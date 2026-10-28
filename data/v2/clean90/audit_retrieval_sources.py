import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "data/v2/clean90"
EXCLUDED = {
    f"src/evaluation/{name}.py"
    for name in ("soft_fuse_dataset", "e2e_manifest", "rag_dataset", "rewriter_noisy_dataset")
}


def normalize_source(source : str)->str:
    #1.只统一来源路径，正文保持原样
    source = PurePosixPath(source.replace("\\", "/")).as_posix()
    return source if source.startswith("src/") else "src/" + source


def audit(path : Path)->dict:
    #1.动作决定检索调用数，返回片段从同序的工具上下文读取
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    totals = Counter(retrieval_calls=0, returned_fragments=0, evaluation_retrieval_calls=0,
                     excluded_retrieval_calls=0, question_verbatim_retrieval_calls=0)
    source_fragments, source_calls = Counter(), Counter()
    evaluation_calls, excluded_calls = Counter(), Counter()
    samples = {}
    for row in rows:
        identity, question = row["sample_id"], row["question"]
        assert identity not in samples, f"Duplicate sample: {path.name} {identity}"
        assert isinstance(question, str) and question, identity
        actions = [item for item in row["tool_actions"] if item["tool_name"] == "search_codebase"]
        contexts = [item for item in row["tool_contexts"] if item["tool_name"] == "search_codebase"]
        assert len(actions) == len(contexts), f"Action/context mismatch: {path.name} {identity}"
        calls = []
        for action, context in zip(actions, contexts):
            payload = json.loads(context["content"])
            assert action["arguments"]["query"] == payload["query"], identity
            fragments = payload["results"]
            assert isinstance(fragments, list), identity
            sources = [normalize_source(fragment["source"]) for fragment in fragments]
            unique_sources = set(sources)
            evaluation = {source for source in unique_sources if source.startswith("src/evaluation/")}
            excluded = unique_sources & EXCLUDED
            verbatim = any(question in fragment["content"] for fragment in fragments)

            #2.同一次检索的同一文件只记一次，片段数另列
            totals.update(retrieval_calls=1, returned_fragments=len(fragments),
                          evaluation_retrieval_calls=int(bool(evaluation)),
                          excluded_retrieval_calls=int(bool(excluded)),
                          question_verbatim_retrieval_calls=int(verbatim))
            source_fragments.update(sources)
            source_calls.update(unique_sources)
            evaluation_calls.update(evaluation)
            excluded_calls.update(excluded)
            calls.append({"query": action["arguments"]["query"], "returned_fragments": len(fragments),
                          "source_files": sorted(unique_sources), "evaluation_files": sorted(evaluation),
                          "excluded_files": sorted(excluded), "question_verbatim_match": verbatim})

        #3.保留逐题证据，供跨索引判分变化表引用
        evaluation = sorted({source for call in calls for source in call["evaluation_files"]})
        excluded = sorted({source for call in calls for source in call["excluded_files"]})
        verbatim_count = sum(call["question_verbatim_match"] for call in calls)
        samples[identity] = {"question": question, "retrieval_calls": len(calls),
                             "evaluation_files": evaluation, "excluded_files": excluded,
                             "retrieved_excluded_file": bool(excluded),
                             "question_verbatim_retrieval_calls": verbatim_count,
                             "question_verbatim_match": bool(verbatim_count), "calls": calls}

    return {"input_path": str(path), "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records": len(rows), **dict(totals),
            "source_fragment_counts": dict(sorted(source_fragments.items())),
            "source_retrieval_counts": dict(sorted(source_calls.items())),
            "evaluation_source_retrieval_counts": dict(sorted(evaluation_calls.items())),
            "excluded_source_retrieval_counts": {source: excluded_calls[source] for source in sorted(EXCLUDED)},
            "samples": samples}


def main():
    #1.同名输出存在就停，不改输入和已有结果
    output = EXP / "retrieval_audit.json"
    assert not output.exists(), f"Refusing overwrite: {output}"
    report = {"started_at": datetime.now(timezone.utc).isoformat(),
              "counting_unit": "search_codebase action; same source counted once per retrieval",
              "source_normalization": "Windows separators normalized; src/ prefix restored",
              "question_match": "Exact original question substring in any returned fragment content",
              "excluded_files": sorted(EXCLUDED), "experiments": {}}
    for label, directory in (("frozen", "frozen90"), ("clean", "clean90")):
        report["experiments"][label] = {
            backend: audit(ROOT / "data/v2" / directory / f"e2e_{backend}.jsonl")
            for backend in ("native", "q4")
        }
    #2.清洁索引若命中排除文件，先保留审计再报错
    clean_ok = all(item["excluded_retrieval_calls"] == 0 for item in report["experiments"]["clean"].values())
    report.update(ended_at=datetime.now(timezone.utc).isoformat(),
                  status="passed" if clean_ok else "failed_clean_retrieved_excluded_file")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    assert clean_ok, "Clean index retrieved excluded exam files; preserved retrieval_audit.json"
    print("Retrieval audit passed:", output)


if __name__ == "__main__":
    main()
