import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import numpy as np

ROOT = Path.cwd()
EXP = Path(__file__).resolve().parent
SOURCE = ROOT / "data/v2/frozen90/source"
REMOVED = (
    "evaluation\\soft_fuse_dataset.py",
    "evaluation\\e2e_manifest.py",
    "evaluation\\rag_dataset.py",
    "evaluation\\rewriter_noisy_dataset.py",
)


def file_hashes(directory : Path)->dict:
    #1.直接读文件，不让 Chroma 打开原索引
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                result[path.relative_to(directory).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def save(report : dict, name : str):
    (EXP / f"index_derivation_{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")


def main():
    #1.旧库只能复制到系统临时目录后再打开
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=("src", "src_ast", "src_flash", "src_ast_flash"), required=True)
    args = parser.parse_args()
    original_path = ROOT / "data/indices" / args.name
    target_path = ROOT / "data/indices" / ("src_frozen90_clean" if args.name == "src" else args.name + "_noexam")
    report = {"name": args.name, "status": "started", "started_at": datetime.now(timezone.utc).isoformat()}
    started = time.perf_counter()
    try:
        assert chromadb.__version__ == "1.5.9", "Chroma version mismatch"
        assert args.name == "src" or not target_path.exists(), "New target already exists"
        before = file_hashes(original_path)
        temporary = Path(tempfile.mkdtemp(prefix=f"soft-fuse-clean-{args.name}-"))
        report.update(original_path=str(original_path), target_path=str(target_path), temporary_directory=str(temporary))
        save(report, args.name)
        shutil.copytree(original_path, temporary / "original")
        assert file_hashes(temporary / "original") == before, "Original copy checksum mismatch"
        source_client = chromadb.PersistentClient(path=str(temporary / "original"))
        source_collection = source_client.get_collection("langchain", embedding_function=None)
        original = source_collection.get(include=["embeddings", "documents", "metadatas"])
        vectors = np.asarray(original["embeddings"])
        count = len(original["ids"])
        assert count > 0 and vectors.shape == (count, 1024), "Vector dimensions mismatch"
        assert len(set(original["ids"])) == count, "Duplicate original ids"
        assert np.isfinite(vectors).all(), "Nonfinite original vectors"
        assert source_collection.configuration["hnsw"]["space"] == "l2", "Original distance is not l2"
        counts = Counter(metadata["source"] for metadata in original["metadatas"])
        keep = [i for i, metadata in enumerate(original["metadatas"]) if metadata["source"] not in REMOVED]
        removed = [{"id": original["ids"][i], "source": metadata["source"]} for i, metadata in enumerate(original["metadatas"]) if metadata["source"] in REMOVED]
        report.update(original_count=count, clean_count=len(keep), removed_count=len(removed), removed=removed, removed_source_counts={source: counts[source] for source in REMOVED})
        save(report, args.name)
        print(json.dumps({"name": args.name, "before": count, "after": len(keep), "removed_source_counts": report["removed_source_counts"]}, ensure_ascii=False), flush=True)

        #2.新增三套库直接复制向量；已有字符标准干净库只验证临时副本
        if args.name == "src":
            clean_before = file_hashes(target_path)
            shutil.copytree(target_path, temporary / "clean")
            assert file_hashes(temporary / "clean") == clean_before, "Clean copy checksum mismatch"
            target_client = chromadb.PersistentClient(path=str(temporary / "clean"))
            clean = target_client.get_collection("langchain", embedding_function=None)
        else:
            target_client = chromadb.PersistentClient(path=str(target_path))
            clean = target_client.create_collection("langchain", embedding_function=None)
            clean.add(ids=[original["ids"][i] for i in keep], embeddings=vectors[keep], documents=[original["documents"][i] for i in keep], metadatas=[original["metadatas"][i] for i in keep])
        actual = clean.get(include=["embeddings", "documents", "metadatas"])
        assert clean.configuration["hnsw"]["space"] == "l2", "Clean distance is not l2"
        assert clean.count() == len(keep), "Clean count mismatch"
        original_by_id = {original["ids"][i]: i for i in keep}
        assert set(actual["ids"]) == set(original_by_id), "Retained ids mismatch"
        max_difference = 0.0
        for j, identity in enumerate(actual["ids"]):
            i = original_by_id[identity]
            assert np.array_equal(actual["embeddings"][j], vectors[i]), "Retained vector mismatch: " + identity
            max_difference = max(max_difference, float(np.max(np.abs(actual["embeddings"][j] - vectors[i]))))
            assert actual["documents"][j] == original["documents"][i], "Retained document mismatch: " + identity
            assert actual["metadatas"][j] == original["metadatas"][i], "Retained metadata mismatch: " + identity
        clean_counts = Counter(metadata["source"] for metadata in actual["metadatas"])
        assert clean_counts == Counter({source: n for source, n in counts.items() if source not in REMOVED}), "Retained source counts mismatch"
        assert all(clean_counts[source] == 0 for source in REMOVED), "Exam blocks remain"

        #3.用冻结 BM25 实现重建，并重新加载核对正文和元数据
        sys.path.insert(0, str(SOURCE))
        from langchain_core.documents import Document
        from src.config import settings
        from src.rag.bm25 import BM25Retiever
        documents = [Document(page_content=original["documents"][i], metadata=original["metadatas"][i]) for i in keep]
        settings.bm25_persist_dir = str((temporary / "clean" if args.name == "src" else target_path) / "bm25_index.pkl")
        if args.name != "src":
            BM25Retiever().index_document(documents)
        bm25 = BM25Retiever()
        bm25.load()
        expected_docs = Counter((doc.page_content, json.dumps(doc.metadata, sort_keys=True)) for doc in documents)
        actual_docs = Counter((doc.page_content, json.dumps(doc.metadata, sort_keys=True)) for doc in bm25.docs)
        assert expected_docs == actual_docs, "BM25 documents mismatch"
        assert len(bm25.search("RepoAgent chat", top_k=6)) == 6, "BM25 smoke failed"
        assert file_hashes(original_path) == before, "Original index changed"
        if args.name == "src":
            assert file_hashes(target_path) == clean_before, "Existing clean index changed"
        norms = np.linalg.norm(vectors[keep], axis=1)
        report.update(status="passed", collection="langchain", distance="l2", dimensions=1024, chromadb_version=chromadb.__version__, embeddings_elementwise_equal=True, max_absolute_embedding_difference=max_difference, documents_equal=True, metadata_equal=True, ids_equal=True, removed_sources_remaining={source: clean_counts[source] for source in REMOVED}, embedding_api_calls_for_derivation=0, bm25_documents_equal=True, bm25_smoke_passed=True, original_index_unchanged=True, retained_vector_norm_min=float(norms.min()), retained_vector_norm_max=float(norms.max()), elapsed_seconds=time.perf_counter()-started, ended_at=datetime.now(timezone.utc).isoformat())
        save(report, args.name)
        print(json.dumps({"name": args.name, "status": report["status"], "max_absolute_embedding_difference": max_difference, "seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error) if isinstance(error, AssertionError) else "See failure type; details suppressed to protect settings", ended_at=datetime.now(timezone.utc).isoformat())
        save(report, args.name)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
