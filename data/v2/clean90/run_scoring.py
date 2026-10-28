import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "data/v2/clean90"
FROZEN = ROOT / "data/v2/frozen90"
SOURCE = FROZEN / "source"
run = json.loads((EXP / "run.json").read_text("utf-8"))
parent = json.loads((FROZEN / "run.json").read_text("utf-8"))
env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
sample_ids = {row["id"] for row in run["manifest"]}


def save(status):
    run.update(status=status, updated_at=datetime.now(timezone.utc).isoformat())
    (EXP / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", "utf-8")


def check_inputs():
    #1.冻结输入只读校验，不打开索引，也不输出密钥
    for path, digest in parent["source_sha256_frozen"].items():
        assert hashlib.sha256((SOURCE / path).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((ROOT / ".env").read_bytes()).hexdigest() == parent["environment_file_sha256"]
    for name in ("native", "q4"):
        entry = next(item for item in run["runs"] if item["name"] == f"generate_{name}")
        path = EXP / f"e2e_{name}.jsonl"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["output_sha256"]
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
        assert len(rows) == 90 and {row["sample_id"] for row in rows} == sample_ids


def validate_output(name, output):
    #1.只检查完整性；正式统计由冻结比较器计算
    if output.suffix == ".jsonl":
        rows = [json.loads(line) for line in output.read_text("utf-8").splitlines() if line.strip()]
        assert len(rows) == 90 and {row["sample_id"] for row in rows} == sample_ids
        digest = hashlib.sha256((EXP / f"e2e_{name.removeprefix('judge_')}.jsonl").read_bytes()).hexdigest()
        for row in rows:
            assert not row.get("error") and row["input_sha256"] == digest
            assert row["judge_model"] == "qwen3.8-max" and row["judge_max_tokens"] == 1200 and row["judge_temperature"] == 0.1
            assert row["judge_gold_mode"] == "legacy" and row["answer_format"] == "full"
        return 90
    result = json.loads(output.read_text("utf-8"))
    assert result["subsets"]["all"]["n"] == 90 and result["subsets"]["remaining68"]["n"] == 68
    assert result["subsets"]["all"]["must_ground_n"] == 35
    assert result["publishable"] and result["review_plan"] is None
    return 90


def execute(name, args, output):
    #1.仅保存通过校验的记录；未通过的本轮输出删除重跑
    log_path = EXP / f"{name}.log"
    assert not output.exists() and not log_path.exists()
    command = [sys.executable, "-B", *args]
    for attempt in range(2):
        check_inputs()
        started = datetime.now(timezone.utc).isoformat()
        save(name)
        try:
            with log_path.open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                run["scoring_process"] = {"runner_pid": os.getpid(), "child_pid": process.pid, "stage": name, "started_at": started}
                save(name)
                assert process.wait() == 0
            records = validate_output(name, output)
        except (OSError, ValueError, KeyError, AssertionError):
            for path in (output, log_path):
                assert path.resolve().parent == EXP.resolve()
                path.unlink(missing_ok=True)
            continue
        run["runs"].append({"name": name, "argv": command, "started_at": started,
                            "ended_at": datetime.now(timezone.utc).isoformat(), "records": records,
                            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
        run.pop("scoring_process", None)
        save(name + "_complete")
        return
    raise RuntimeError("Scoring incomplete")


def main():
    #1.按原顺序评分并比较，参数与冻结实验相同
    for name in ("native", "q4"):
        output = EXP / f"judge_{name}.jsonl"
        execute(f"judge_{name}", [str(SOURCE / "scripts/judge_answers.py"), "--jsonl", str(EXP / f"e2e_{name}.jsonl"), "--full", "--out", str(output), "--judge-model", "qwen3.8-max", "--max-tokens", "1200", "--temperature", "0.1", "--workers", "4"], output)
    execute("compare", [str(SOURCE / "scripts/eval_agent.py"), "--compare", str(EXP / "e2e_native.jsonl"), str(EXP / "e2e_q4.jsonl"), "--judges", str(EXP / "judge_native.jsonl"), str(EXP / "judge_q4.jsonl"), "--compare-out", str(EXP / "comparison.json")], EXP / "comparison.json")
    check_inputs()
    run["scoring_runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save("scoring_and_comparison_complete_summary_pending")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        run.pop("scoring_process", None)
        save("scoring_incomplete")
