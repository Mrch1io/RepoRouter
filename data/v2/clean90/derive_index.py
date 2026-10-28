import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import numpy as np

ROOT = Path.cwd()
EXP = ROOT / "data/v2/clean90"
SOURCE = ROOT / "data/v2/frozen90/source"
ORIGINAL = ROOT / "data/indices/src"
TARGET = ROOT / "data/indices/src_frozen90_clean"
parent = json.loads((SOURCE.parent / "run.json").read_text("utf-8"))
report = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "started"}
removed_sources = {
    "evaluation\\soft_fuse_dataset.py": 6,
    "evaluation\\e2e_manifest.py": 3,
    "evaluation\\rag_dataset.py": 3,
    "evaluation\\rewriter_noisy_dataset.py": 3,
}
retained_evaluation = {
    "evaluation\\llm_judge_e2e.py": 7,
    "evaluation\\eva_retrieval.py": 4,
    "evaluation\\eva_retrieval_reranker.py": 3,
    "evaluation\\eva_rewriter.py": 3,
    "evaluation\\chroma_verify_tokens.py": 1,
    "evaluation\\stats.py": 1,
}


def logical_hashes(index):
    #1.只读连接，不让 Chroma 打开原索引
    with sqlite3.connect((index / "chroma.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        return {
            table: hashlib.sha256(repr(sorted(db.execute(f'SELECT * FROM "{table}"').fetchall(), key=repr)).encode()).hexdigest()
            for table in parent["index_logical_sha256_before"]
        }


def save():
    (EXP / "index_derivation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")


try:
    #1.验证冻结条件，整个旧目录只复制到系统临时目录
    assert chromadb.__version__ == "1.5.9", chromadb.__version__
    assert not any(TARGET.iterdir()), "Clean index directory is not empty"
    assert logical_hashes(ORIGINAL) == parent["index_logical_sha256_before"], "Original logical hashes differ"
    assert hashlib.sha256((ORIGINAL / "bm25_index.pkl").read_bytes()).hexdigest() == parent["index_files_sha256_before"]["bm25_index.pkl"]
    assert hashlib.sha256((ROOT / ".env").read_bytes()).hexdigest() == parent["environment_file_sha256"]
    for path, digest in parent["source_sha256_frozen"].items():
        assert hashlib.sha256((SOURCE / path).read_bytes()).hexdigest() == digest, path
    temporary = Path(tempfile.mkdtemp(prefix="clean90-source-"))
    report["temporary_source_copy"] = str(temporary / "src")
    save()
    shutil.copytree(ORIGINAL, temporary / "src")
    source_client = chromadb.PersistentClient(path=str(temporary / "src"))
    source_collection = source_client.get_collection("langchain", embedding_function=None)
    original = source_collection.get(include=["embeddings", "documents", "metadatas"])
    vectors = np.asarray(original["embeddings"])
    assert vectors.shape == (87, 1024), vectors.shape
    assert len(set(original["ids"])) == 87
    assert source_collection.configuration["hnsw"]["space"] == "l2"
    counts = Counter(metadata["source"] for metadata in original["metadatas"])
    assert {source: counts[source] for source in removed_sources} == removed_sources
    assert {source: counts[source] for source in retained_evaluation} == retained_evaluation
    keep = [i for i, metadata in enumerate(original["metadatas"]) if metadata["source"] not in removed_sources]
    removed = [{"id": original["ids"][i], "source": metadata["source"]} for i, metadata in enumerate(original["metadatas"]) if metadata["source"] in removed_sources]
    assert len(keep) == 72 and len(removed) == 15

    #2.原样写入向量、正文和元数据，不调用嵌入 API
    target_client = chromadb.PersistentClient(path=str(TARGET))
    clean = target_client.create_collection("langchain", embedding_function=None)
    clean.add(ids=[original["ids"][i] for i in keep], embeddings=vectors[keep], documents=[original["documents"][i] for i in keep], metadatas=[original["metadatas"][i] for i in keep])
    actual = clean.get(include=["embeddings", "documents", "metadatas"])
    assert clean.configuration["hnsw"]["space"] == "l2"
    assert clean.count() == 72
    original_by_id = {original["ids"][i]: i for i in keep}
    assert set(actual["ids"]) == set(original_by_id)
    max_difference = 0.0
    for j, identity in enumerate(actual["ids"]):
        i = original_by_id[identity]
        assert np.array_equal(actual["embeddings"][j], vectors[i]), identity
        max_difference = max(max_difference, float(np.max(np.abs(actual["embeddings"][j] - vectors[i]))))
        assert actual["documents"][j] == original["documents"][i], identity
        assert actual["metadatas"][j] == original["metadatas"][i], identity
    clean_counts = Counter(metadata["source"] for metadata in actual["metadatas"])
    assert all(clean_counts[source] == 0 for source in removed_sources)
    assert {source: clean_counts[source] for source in retained_evaluation} == retained_evaluation
    report.update(original_count=87, clean_count=72, dimensions=1024, collection="langchain", distance="l2", chromadb_version=chromadb.__version__, removed=removed, removed_source_counts=removed_sources, removed_sources_remaining={source: clean_counts[source] for source in removed_sources}, retained_evaluation_counts=retained_evaluation, embeddings_elementwise_equal=True, max_absolute_embedding_difference=max_difference, documents_equal=True, metadata_equal=True, embedding_api_calls_for_derivation=0)
    save()

    #3.用冻结 BM25 实现建词频索引，再用冻结 VectorStore 冒烟一次
    os.environ["Chroma_persist_dir"] = "./data/indices/src_frozen90_clean"
    os.environ["bm25_persist_dir"] = "./data/indices/src_frozen90_clean/bm25_index.pkl"
    sys.path.insert(0, str(SOURCE))
    from langchain_core.documents import Document
    from src.rag.bm25 import BM25Retiever
    from src.rag.vectorstore import VectorStore
    documents = [Document(page_content=original["documents"][i], metadata=original["metadatas"][i]) for i in keep]
    BM25Retiever().index_document(documents)
    store = VectorStore()
    store.load()
    started = time.perf_counter()
    hits = store.search_dense("RepoAgent.chat 在哪里实现", top_k=6)
    assert len(hits) == 6
    assert all(hit.metadata["source"] not in removed_sources for hit in hits)
    report["dense_smoke"] = {"query": "RepoAgent.chat 在哪里实现", "top_k": 6, "returned": len(hits), "sources": [hit.metadata["source"] for hit in hits], "seconds": time.perf_counter() - started, "status": "passed"}
    assert logical_hashes(ORIGINAL) == parent["index_logical_sha256_before"]
    assert hashlib.sha256((ORIGINAL / "bm25_index.pkl").read_bytes()).hexdigest() == parent["index_files_sha256_before"]["bm25_index.pkl"]
    report.update(status="derivation_and_smoke_passed", ended_at=datetime.now(timezone.utc).isoformat(), original_index_unchanged=True)
    save()
    print(json.dumps({"status": report["status"], "counts": [87, 72], "removed": removed, "max_absolute_embedding_difference": max_difference, "dense_smoke": report["dense_smoke"]}, ensure_ascii=False), flush=True)
except Exception as error:
    report.update(status="failed_preserve_records", error=f"{type(error).__name__}: {error}", ended_at=datetime.now(timezone.utc).isoformat())
    save()
    raise
