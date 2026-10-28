import argparse
import ast
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
EXP = Path(__file__).resolve().parent
SOURCE = ROOT / "data/v2/soft_fuse_clean/source"
NAMES = ("src", "src_ast", "src_flash", "src_ast_flash")
CLEAN = ("src_frozen90_clean", "src_ast_noexam", "src_flash_noexam", "src_ast_flash_noexam")
RESTORED = {"p02", "p21", "p23", "p29", "m01", "m04"}
SPLITS = {"domain": 40, "far_general": 28, "term_near_no_rag": 12, "misleading_edge": 10}
VARIABLES = dict(zip(("DOMAIN_IN_PROJECT", "OUTDOMAIN_FAR_GENERAL", "OUTDOMAIN_TERM_NEAR_NO_RAG", "OUTDOMAIN_MISLEADING_EDGE"), SPLITS))
REMOVED = {"evaluation/soft_fuse_dataset.py", "evaluation/e2e_manifest.py", "evaluation/rag_dataset.py", "evaluation/rewriter_noisy_dataset.py"}
DATASET = "src/evaluation/soft_fuse_dataset.py"


def read_json(path : Path)->dict:
    return json.loads(path.read_text("utf-8"))


def save(name : str, payload : dict):
    (EXP / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")


def file_hashes(directory : Path)->dict:
    #1.沿用派生实验的逐文件摘要，只读文件，不让 Chroma 打开原索引
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                result[path.relative_to(directory).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def check_protected()->dict:
    #1.对保护目录完整枚举，同时检查文件丢失、新增和内容变化
    before = read_json(EXP / "protected_before.json")
    actual = {}
    for relative in before["roots"]:
        path = ROOT / relative
        paths = sorted(path.rglob("*")) if path.is_dir() else [path]
        for file in paths:
            if file.is_file():
                with file.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                actual[file.relative_to(ROOT).as_posix()] = {"bytes": file.stat().st_size, "sha256": digest}
    old = before["files"]
    changed = [{"path": path, "before": old[path], "after": actual[path]} for path in old.keys() & actual.keys() if old[path] != actual[path]]
    missing = sorted(old.keys() - actual.keys())
    added = sorted(actual.keys() - old.keys())
    return {"status": "passed" if not changed and not missing and not added else "failed", "protected_file_count": len(old), "changed": changed, "missing": missing, "added": added, "files": actual}


def dataset_rows(path : Path)->dict:
    #1.只解析四个题目常量，不执行数据文件
    rows = {}
    for node in ast.parse(path.read_text("utf-8")).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in VARIABLES:
            split = VARIABLES[node.target.id]
            for item in ast.literal_eval(node.value):
                key = (split, item["id"])
                assert key not in rows, "Duplicate dataset id"
                rows[key] = item["query"]
    assert Counter(key[0] for key in rows) == SPLITS, "Dataset split counts mismatch"
    return rows


def validate_rows(payload : dict, expected : dict)->dict:
    #1.检查完整题集、题面和有限距离，避免漏题后仍输出统计
    rows = payload["rows"]
    assert Counter(row["split"] for row in rows) == SPLITS, "Split counts mismatch"
    keyed = {(row["split"], row["id"]): row for row in rows}
    assert len(keyed) == 90 and keyed.keys() == expected.keys(), "Question ids mismatch"
    assert all(row["query"] == expected[key] for key, row in keyed.items()), "Query text mismatch"
    assert all(math.isfinite(row["top1_distance"]) for row in rows), "Nonfinite distance"
    return keyed


def restore_queries(source : Path, expected : dict, historical_name : str)->dict:
    if not historical_name.strip() or historical_name.casefold() == "repoagent":
        raise ValueError("历史题面名称不能是空白或当前名称 RepoAgent；请核对修改前备份")
    #1.仅在临时副本的六行题面中还原名称，保留原换行和其他字节
    path = source / DATASET
    lines = path.read_bytes().decode("utf-8").splitlines(keepends=True)
    changed = []
    for i, line in enumerate(lines):
        for identity in sorted(RESTORED):
            if f'"id": "{identity}"' in line:
                assert '"query":' in line and "RepoAgent" in line, "Restoration line mismatch"
                changed.append({"id": identity, "occurrences": line.count("RepoAgent")})
                lines[i] = line.replace("RepoAgent", historical_name)
    assert {row["id"] for row in changed} == RESTORED and len(changed) == 6, "Restoration ids mismatch"
    assert sum(row["occurrences"] for row in changed) == 7, "Restoration count mismatch"
    path.write_bytes("".join(lines).encode("utf-8"))
    restored = dataset_rows(path)
    intended = {key: query.replace("RepoAgent", historical_name) if key[1] in RESTORED else query for key, query in expected.items()}
    assert restored == intended, "Restored dataset mismatch"
    rows = [{"split": key[0], "id": key[1], "original_query": expected[key], "restored_query": restored[key]} for key in expected if key[1] in RESTORED]
    save("question_restoration.json", {"status": "passed", "file": str(path), "changed_questions": rows, "replacement_count": 7})
    return restored


def compare(old : dict, new : dict)->dict:
    #1.历史 JSON 的题面被替换过，所以按题号对齐，只验收指定三项
    left = {(row["split"], row["id"]): row for row in old["rows"]}
    right = {(row["split"], row["id"]): row for row in new["rows"]}
    assert len(left) == len(right) == 90 and left.keys() == right.keys(), "Comparison ids mismatch"
    assert old["embedding"] == new["embedding"], "Embedding mismatch"
    differences = []
    for key, row in left.items():
        actual = right[key]
        differences.append({"split": key[0], "id": key[1], "historical_json_query": row["query"], "restored_query": actual["query"], "old_source": row["top1_source"], "new_source": actual["top1_source"], "source_equal": row["top1_source"] == actual["top1_source"], "old_distance": row["top1_distance"], "new_distance": actual["top1_distance"], "absolute_distance_difference": abs(row["top1_distance"] - actual["top1_distance"])})
    result = {"source_matches": sum(row["source_equal"] for row in differences), "max_absolute_distance_difference": max(row["absolute_distance_difference"] for row in differences), "old_auc": old["roc_auc"], "new_auc": new["roc_auc"], "old_auc_3dp": format(old["roc_auc"], ".3f"), "new_auc_3dp": format(new["roc_auc"], ".3f"), "rows": differences}
    result["passed"] = result["source_matches"] == 90 and result["max_absolute_distance_difference"] < 1e-3 and result["old_auc_3dp"] == result["new_auc_3dp"]
    return result


def evaluate(source : Path, index_name : str, output_name : str, embedding : str, expected : dict, report : dict)->dict:
    #1.每次都复制索引并核对摘要，Chroma 只接触系统临时目录
    original = ROOT / "data/indices" / index_name
    assert (original / "chroma.sqlite3").is_file(), "Index missing: " + index_name
    before = file_hashes(original)
    source_before = file_hashes(source)
    temporary = Path(tempfile.mkdtemp(prefix=f"soft-fuse-rebaseline-{index_name}-"))
    shutil.copytree(original, temporary / "index")
    assert file_hashes(temporary / "index") == before, "Index copy mismatch: " + index_name
    output, log = EXP / f"{output_name}.json", EXP / f"{output_name}.log"
    assert not output.exists() and not log.exists(), "Evaluation output already exists"
    command = [sys.executable, "-B", str(source / "scripts/eval_soft_fuse.py"), "--persist-dir", str(temporary / "index"), "--embedding", embedding, "--max-false-block", "0.075", "--json-out", str(output)]
    run = {"name": output_name, "index": str(original), "temporary_directory": str(temporary), "source": str(source), "command": command, "cwd": str(ROOT), "output": str(output), "log": str(log), "status": "running"}
    report["runs"].append(run)
    save("run.json", report)
    print(json.dumps({"name": output_name, "status": "running"}), flush=True)
    #2.仅调用原评估脚本做查询嵌入，禁写字节码，工作目录仍是项目根目录
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    run.update(elapsed_seconds=time.perf_counter()-started, returncode=completed.returncode)
    assert completed.returncode == 0, "Evaluation process failed: " + output_name
    assert file_hashes(original) == before, "Original index changed: " + index_name
    assert file_hashes(source) == source_before, "Evaluation source changed"
    payload = read_json(output)
    validate_rows(payload, expected)
    assert payload["embedding"] == embedding, "Output embedding mismatch"
    if index_name in CLEAN:
        assert all(row["top1_source"].replace("\\", "/").removeprefix("src/") not in REMOVED for row in payload["rows"]), "Exam source remains in clean result"
    run.update(status="passed", original_index_unchanged=True, source_unchanged=True)
    save("run.json", report)
    print(json.dumps({"name": output_name, "status": "passed", "seconds": run["elapsed_seconds"], "roc_auc": payload["roc_auc"], "recommended": payload["recommended"]}), flush=True)
    return payload


def main():
    if (EXP / "run.json").exists():
        raise SystemExit("STOP: experiment already started; existing records preserved")
    parser = argparse.ArgumentParser(description="复现软熔断标定，需配套的原始源码、索引和校验清单")
    parser.add_argument("--historical-name", required=True, help="修改前备份中记录的原始项目名")
    args = parser.parse_args()
    report = {"status": "started", "started_at": datetime.now(timezone.utc).isoformat(), "runs": []}
    try:
        #1.复用旧脚本结构，先复制冻结源码，再只修改解释性副本
        assert Path(sys.executable).resolve() == (ROOT / ".venv/Scripts/python.exe").resolve(), "Wrong Python interpreter"
        assert check_protected()["status"] == "passed", "Protected files changed since initial snapshot"
        temporary = Path(tempfile.mkdtemp(prefix="soft-fuse-rebaseline-source-"))
        report["temporary_source_directory"] = str(temporary)
        source_hashes = file_hashes(SOURCE)
        baseline_source, explain_source = temporary / "baseline_source", temporary / "explain_source"
        for copy in (baseline_source, explain_source):
            shutil.copytree(SOURCE, copy)
            assert file_hashes(copy) == source_hashes, "Frozen source copy mismatch"
        report["source_sha256"] = source_hashes
        expected = dataset_rows(baseline_source / DATASET)
        restored = restore_queries(explain_source, expected, args.historical_name)
        changed_hashes = file_hashes(explain_source)
        assert [path for path in source_hashes if source_hashes[path] != changed_hashes[path]] == [DATASET], "Unexpected source modification"
        historical = {name: read_json(ROOT / f"data/v2/soft_fuse_{name}.json") for name in NAMES}
        for old in historical.values():
            validate_rows(old, expected)
        save("run.json", report)
        #2.解释性核对不通过就停，不能进入重新基线
        payload = evaluate(explain_source, "src", "explain_src", historical["src"]["embedding"], restored, report)
        comparison = compare(historical["src"], payload)
        save("explanation_check.json", comparison)
        assert comparison["passed"], "Explanation comparison failed"
        report["explanation_passed"] = True
        integrity = check_protected()
        save("protected_after_explanation.json", integrity)
        assert integrity["status"] == "passed", "Protected files changed during explanation"
        print(json.dumps({"stage": "explanation", "status": "passed", "source_matches": comparison["source_matches"], "max_absolute_distance_difference": comparison["max_absolute_distance_difference"], "auc_3dp": comparison["new_auc_3dp"]}), flush=True)
        #3.原样 RepoAgent 题面跑四套原索引和四套已有干净索引
        for indices in (NAMES, CLEAN):
            for original_name, index_name in zip(NAMES, indices):
                assert file_hashes(baseline_source) == source_hashes, "Baseline source mismatch"
                evaluate(baseline_source, index_name, "baseline_" + index_name, historical[original_name]["embedding"], expected, report)
        report.update(status="passed", ended_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error) if isinstance(error, AssertionError) else "Details suppressed to protect settings", ended_at=datetime.now(timezone.utc).isoformat())
        if report["runs"] and report["runs"][-1]["status"] == "running":
            report["runs"][-1]["status"] = "failed"
    #4.结束时再次逐文件核对保护范围，失败记录保留，不重跑
    integrity = check_protected()
    save("protected_after.json", integrity)
    report["protected_integrity"] = {key: value for key, value in integrity.items() if key != "files"}
    if integrity["status"] != "passed":
        report["status"] = "failed"
        report["integrity_error"] = "Protected files changed"
    save("run.json", report)
    print(json.dumps({"status": report["status"], "error": report.get("error"), "protected_integrity": report["protected_integrity"]}), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
