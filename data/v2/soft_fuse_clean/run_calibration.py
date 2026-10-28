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

from derive_index import file_hashes, REMOVED

ROOT = Path.cwd()
EXP = Path(__file__).resolve().parent
NAMES = ("src", "src_ast", "src_flash", "src_ast_flash")
SPLITS = {"domain": 40, "far_general": 28, "term_near_no_rag": 12, "misleading_edge": 10}


def read_json(path : Path)->dict:
    return json.loads(path.read_text("utf-8"))


def save(name : str, payload : dict):
    (EXP / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")


def validate_rows(payload : dict)->dict:
    #1.按题号和分组对齐，不能把缺题或非有限距离当成有效结果
    rows = payload["rows"]
    assert Counter(row["split"] for row in rows) == SPLITS, "Split counts mismatch"
    keyed = {(row["split"], row["id"]): row for row in rows}
    assert len(keyed) == 90, "Duplicate question ids"
    assert all(math.isfinite(row["top1_distance"]) for row in rows), "Nonfinite distance"
    return keyed


def compare(old : dict, new : dict)->dict:
    #1.保留逐题差异，失败后直接终止，不调整条件或重跑
    left, right = validate_rows(old), validate_rows(new)
    assert old["embedding"] == new["embedding"], "Embedding mismatch"
    assert left.keys() == right.keys(), "Question ids mismatch"
    assert all(left[key]["query"] == right[key]["query"] for key in left), "Query text mismatch"
    differences = []
    for key, row in left.items():
        actual = right[key]
        delta = abs(row["top1_distance"] - actual["top1_distance"])
        differences.append({"split": key[0], "id": key[1], "query": row["query"], "old_source": row["top1_source"], "new_source": actual["top1_source"], "source_equal": row["top1_source"] == actual["top1_source"], "old_distance": row["top1_distance"], "new_distance": actual["top1_distance"], "absolute_distance_difference": delta})
    result = {"source_mismatches": [row for row in differences if not row["source_equal"]], "max_absolute_distance_difference": max(row["absolute_distance_difference"] for row in differences), "old_auc_3dp": format(old["roc_auc"], ".3f"), "new_auc_3dp": format(new["roc_auc"], ".3f"), "old_threshold_3dp": format(old["recommended"]["threshold"], ".3f"), "new_threshold_3dp": format(new["recommended"]["threshold"], ".3f"), "rows": differences}
    result["passed"] = not result["source_mismatches"] and result["max_absolute_distance_difference"] < 1e-3 and result["old_auc_3dp"] == result["new_auc_3dp"] and result["old_threshold_3dp"] == result["new_threshold_3dp"]
    return result


def main():
    report = {"status": "started", "started_at": datetime.now(timezone.utc).isoformat(), "runs": []}
    try:
        #1.复制冻结代码和当前评估脚本，原文件全部只读
        assert Path(sys.executable).resolve() == (ROOT / ".venv/Scripts/python.exe").resolve(), "Wrong Python interpreter"
        for name in NAMES:
            assert read_json(EXP / f"index_derivation_{name}.json")["status"] == "passed", "Derivation incomplete: " + name
        frozen = ROOT / "data/v2/frozen90/source"
        assert not (EXP / "source").exists(), "Experiment source already exists"
        shutil.copytree(frozen, EXP / "source")
        assert file_hashes(frozen) == file_hashes(EXP / "source"), "Frozen copy mismatch"
        script = EXP / "source/scripts/eval_soft_fuse.py"
        assert not script.exists(), "Eval script target already exists"
        shutil.copy2(ROOT / "scripts/eval_soft_fuse.py", script)
        assert script.read_bytes() == (ROOT / "scripts/eval_soft_fuse.py").read_bytes(), "Eval script copy mismatch"
        report["eval_script_sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
        save("calibration_run.json", report)

        #2.四套原索引依次复现；全部通过后才运行干净索引
        for stage in ("reproduce", "clean"):
            for name in NAMES:
                old = read_json(ROOT / f"data/v2/soft_fuse_{name}.json")
                index_name = name if stage == "reproduce" else ("src_frozen90_clean" if name == "src" else name + "_noexam")
                original = ROOT / "data/indices" / index_name
                temporary = Path(tempfile.mkdtemp(prefix=f"soft-fuse-{stage}-{name}-"))
                before = file_hashes(original)
                shutil.copytree(original, temporary / "index")
                assert file_hashes(temporary / "index") == before, "Index copy mismatch: " + index_name
                output = EXP / (f"reproduce_{name}.json" if stage == "reproduce" else f"soft_fuse_{name}_clean.json")
                log = output.with_suffix(".log")
                assert not output.exists() and not log.exists(), "Calibration output already exists"
                command = [sys.executable, "-B", str(script), "--persist-dir", str(temporary / "index"), "--embedding", old["embedding"], "--max-false-block", "0.075", "--json-out", str(output)]
                run = {"stage": stage, "name": name, "status": "running", "temporary_directory": str(temporary), "index": str(original), "command": command, "cwd": str(ROOT), "output": str(output), "log": str(log)}
                report["runs"].append(run)
                save("calibration_run.json", report)
                print(json.dumps({"stage": stage, "name": name, "status": "running"}), flush=True)
                started = time.perf_counter()
                environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
                with log.open("w", encoding="utf-8") as stream:
                    completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                run.update(elapsed_seconds=time.perf_counter()-started, returncode=completed.returncode)
                assert completed.returncode == 0, f"Evaluation process failed: {stage}/{name}; see log"
                assert file_hashes(original) == before, "Original index changed: " + index_name
                new = read_json(output)
                actual_rows = validate_rows(new)
                expected_rows = validate_rows(old)
                assert actual_rows.keys() == expected_rows.keys(), "Question ids mismatch"
                assert all(row["query"] == expected_rows[key]["query"] for key, row in actual_rows.items()), "Query mismatch"
                assert new["embedding"] == old["embedding"], "Embedding mismatch"
                if stage == "reproduce":
                    result = compare(old, new)
                    save(f"reproduction_check_{name}.json", result)
                    assert result["passed"], "Old result reproduction mismatch: " + name
                    run["max_absolute_distance_difference"] = result["max_absolute_distance_difference"]
                else:
                    assert all(row["top1_source"] not in REMOVED for row in new["rows"]), "Exam source remains in clean results"
                run.update(status="passed", original_index_unchanged=True)
                save("calibration_run.json", report)
                print(json.dumps({"stage": stage, "name": name, "status": "passed", "seconds": run["elapsed_seconds"], "roc_auc": new["roc_auc"], "recommended": new["recommended"]}), flush=True)
        report.update(status="passed", ended_at=datetime.now(timezone.utc).isoformat())
        save("calibration_run.json", report)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error) if isinstance(error, AssertionError) else "Details suppressed to protect settings", ended_at=datetime.now(timezone.utc).isoformat())
        save("calibration_run.json", report)
        print(json.dumps({"status": "failed", "error_type": report["error_type"], "error": report["error"]}), flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
