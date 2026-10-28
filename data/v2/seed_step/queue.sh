#!/usr/bin/env bash
set -euo pipefail

RUN=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=/root/autodl-tmp/devagent
PY="$ROOT/venv/bin/python"
BASE=/root/autodl-tmp/models/Qwen/Qwen3.5-4B
LOCK=/root/autodl-tmp/gpu.lock
CURRENT="$RUN"
ITEM=preflight
PHASE=checking
TEST=""
OWN_LOCK=0
JOBS=(v1_240steps_s42 v1_s43 v1_s44 abl_s43 abl_s44 v2_s43 v2_s44)

status() {
    #1.同目录原子替换，查看进度时不会读到半个JSON
    printf '{"status":"%s","stage":"%s","item":"%s","test":"%s","updated_at_utc":"%s"}\n' \
        "$2" "$PHASE" "$ITEM" "$TEST" "$(date -u +%FT%TZ)" > "$1/status.json.tmp"
    mv -- "$1/status.json.tmp" "$1/status.json"
}

read_status() {
    "$PY" -B - "$1/status.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
print(json.loads(path.read_text())["status"] if path.exists() else "pending")
PY
}

gpu_idle() {
    local apps
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
    [[ -z "$apps" ]] || { printf 'GPU is busy: %s\n' "$apps" >&2; return 1; }
}

finish() {
    local code=$?
    trap - EXIT
    if (( code != 0 )); then
        status "$CURRENT" failed
        if [[ "$CURRENT" != "$RUN" ]]; then status "$RUN" failed; fi
    fi
    if (( OWN_LOCK )); then : > "$LOCK"; fi
    exit "$code"
}

#1.先拿锁再检查，另一个启动进程不能改正在运行的状态
exec 9<>"$LOCK"
flock -n 9
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
[[ ! -s "$LOCK" ]] || { printf 'GPU lock contains an existing owner.\n' >&2; exit 1; }
gpu_idle
OWN_LOCK=1
printf 'RepoAgent seed/step queue, PID %s, %s\n' "$$" "$RUN" > "$LOCK"
printf '%s\n' "$$" > "$RUN/runner.pid"

#2.缓存、临时文件和训练相对路径都留在本次目录
export PATH="$ROOT/venv/bin:$PATH"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$RUN/cache"
export HF_HOME="$RUN/cache/huggingface"
export HF_DATASETS_CACHE="$RUN/cache/datasets"
export HF_MODULES_CACHE="$RUN/cache/modules"
export TORCH_HOME="$RUN/cache/torch"
export TRITON_CACHE_DIR="$RUN/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUN/cache/inductor"
export CUDA_CACHE_PATH="$RUN/cache/cuda"
export MPLCONFIGDIR="$RUN/cache/matplotlib"
export TMPDIR="$RUN/tmp"
mkdir -p -- "$TMPDIR" "$XDG_CACHE_HOME"
cd "$RUN"
sha256sum --check inputs.sha256
sha256sum --check adapters.sha256
"$PY" -B - "$RUN" <<'PY'
import importlib.metadata
import json
import os
import sys
from pathlib import Path
run = Path(sys.argv[1])
expected = json.loads((run / "run.json").read_text())["preflight"]["checks"]["package_versions"]["detail"]
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"Package versions changed: {actual}")
disk = os.statvfs(run)
if disk.f_bavail * disk.f_frsize < 20 * 1024**3:
    raise SystemExit("Less than 20 GiB free before queue launch.")
PY

#3.启动前拒绝任何未完成实验的已有adapter
for name in "${JOBS[@]}"; do
    saved=$(read_status "$RUN/$name")
    if [[ "$saved" != complete && -e "$RUN/$name/adapter" ]]; then
        printf 'Incomplete run already has an adapter: %s\n' "$name" >&2
        exit 1
    fi
done

evaluate() {
    local dest=$1 adapter=$2 data=$3 condition=$4
    local tests=(eval_in_domain_no_summary eval_cross_repo_no_summary)
    if [[ "$condition" == v2 ]]; then tests+=(eval_in_domain eval_cross_repo); fi
    for TEST in "${tests[@]}"; do
        gpu_idle
        PHASE=evaluating
        status "$dest" running
        status "$RUN" running
        printf '%s evaluating %s/%s\n' "$(date -u +%FT%TZ)" "$ITEM" "$TEST"
        cd "$RUN/source"
        "$PY" -B loRA/eval_routing_lora.py \
            --base-model "$BASE" --lora-path "$adapter" \
            --test-data "$data/$TEST.json" \
            --max-new-tokens 512 --temperature 0.1 > "$dest/$TEST.log" 2>&1
        "$PY" -B - "$dest" "$TEST" "$condition" <<'PY'
import json
import re
import sys
from pathlib import Path

#1.只取decision_type指标；损坏金标、截断日志都不能当成成功
dest, test, condition = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
text = (dest / f"{test}.log").read_text("utf-8")
def number(pattern):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise SystemExit(f"Expected one metric for {pattern}, got {len(matches)}")
    return int(matches[0])
total = number(r"^样本总数（文件条数）: (\d+)$")
evaluated = number(r"^金标可解析条数: (\d+)（")
skipped = number(r"^金标可解析条数: \d+（跳过金标损坏: (\d+)）$")
correct = number(r"^decision_type 一致: (\d+) \(")
expected = 50 if "in_domain" in test else 87
if total != expected or evaluated != expected or skipped != 0 or not 0 <= correct <= total:
    raise SystemExit(f"Invalid evaluation totals: {total}, {evaluated}, {skipped}, {correct}")
if "参数合法率: N/A" in text:
    raise SystemExit("Frozen tools_schemas.py did not load.")
result = {"test": test, "total": total, "evaluated": evaluated, "skipped_bad_gold": skipped, "decision_type_match": correct, "accuracy": correct / total}
#2.复现有差异只记录，不用历史分数替换今天的分数
if dest.parent.name == "reproduce_s42":
    history = {
        "v1": {"eval_in_domain_no_summary": 49, "eval_cross_repo_no_summary": 71},
        "abl": {"eval_in_domain_no_summary": 49, "eval_cross_repo_no_summary": 78},
        "v2": {"eval_cross_repo_no_summary": 81, "eval_in_domain": 48, "eval_cross_repo": 82},
    }
    previous = history[condition].get(test)
    result.update(historical_match=previous, difference=None if previous is None else correct - previous)
path = dest / "results.json"
results = json.loads(path.read_text("utf-8")) if path.exists() else {}
results[test] = result
temp = dest / "results.json.tmp"
temp.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temp.replace(path)
print(json.dumps(result, ensure_ascii=False), flush=True)
PY
    done
    TEST=""
}

#4.先复现三个seed42，完成的子目录可以跳过
for condition in v1 abl v2; do
    ITEM="reproduce_s42/$condition"
    CURRENT="$RUN/$ITEM"
    saved=$(read_status "$CURRENT")
    if [[ "$saved" == complete ]]; then printf 'Skip complete: %s\n' "$ITEM"; continue; fi
    for previous in "$CURRENT"/eval_*.log "$CURRENT/results.json"; do
        [[ ! -e "$previous" ]] || { printf 'Incomplete reproduction has existing results: %s\n' "$previous" >&2; exit 1; }
    done
    (cd "$CURRENT" && sha256sum --check inputs.sha256)
    adapter=$("$PY" -B - "$RUN/run.json" "$condition" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text())["preflight"]["adapters"][sys.argv[2]]["path"])
PY
)
    evaluate "$CURRENT" "$adapter" "$RUN/data" "$condition"
    PHASE=complete
    status "$CURRENT" complete
done

#5.再按固定顺序训练，步数和epoch都由训练产物核查
for name in "${JOBS[@]}"; do
    ITEM="$name"
    CURRENT="$RUN/$name"
    saved=$(read_status "$CURRENT")
    if [[ "$saved" == complete ]]; then printf 'Skip complete: %s\n' "$ITEM"; continue; fi
    [[ ! -e "$CURRENT/adapter" ]] || { printf 'Adapter already exists: %s\n' "$CURRENT" >&2; exit 1; }
    (cd "$CURRENT" && sha256sum --check inputs.sha256)
    gpu_idle
    TEST=""
    PHASE=training
    status "$CURRENT" running
    status "$RUN" running
    printf '%s training %s\n' "$(date -u +%FT%TZ)" "$name"
    cd "$CURRENT"
    "$ROOT/venv/bin/llamafactory-cli" train "$CURRENT/train.yaml" > "$CURRENT/train.log" 2>&1
    "$PY" -B - "$RUN" "$name" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
run, name = Path(sys.argv[1]), sys.argv[2]
job = next(job for job in json.loads((run / "run.json").read_text())["jobs"] if job["name"] == name)
adapter = run / name / "adapter"
results = json.loads((adapter / "train_results.json").read_text())
state = json.loads((adapter / "trainer_state.json").read_text())
if state["global_step"] != job["expected_steps"]:
    raise SystemExit(f"Unexpected global_step: {state['global_step']}")
if job["expected_epoch"] is not None and (results["epoch"] != 3.0 or state["epoch"] != 3.0):
    raise SystemExit(f"Unexpected epoch: {results['epoch']}, {state['epoch']}")
weights = adapter / "adapter_model.safetensors"
if not weights.is_file() or weights.stat().st_size == 0:
    raise SystemExit("Trained adapter is missing or empty.")
with weights.open("rb") as f:
    digest = hashlib.file_digest(f, "sha256").hexdigest()
check = {"global_step": state["global_step"], "epoch": results["epoch"], "trainer_state_epoch": state["epoch"], "adapter_sha256": digest}
(run / name / "training_check.json").write_text(json.dumps(check, indent=2) + "\n")
print(json.dumps(check), flush=True)
PY
    condition=${name%%_*}
    evaluate "$CURRENT" "$CURRENT/adapter" "$CURRENT/data" "$condition"
    PHASE=complete
    status "$CURRENT" complete
done

#6.汇总只引用今天的复现和本轮训练；步数对照独立列出
CURRENT="$RUN"
ITEM=summary
PHASE=summarizing
status "$RUN" running
"$PY" -B - "$RUN" <<'PY'
import json
import statistics
import sys
from pathlib import Path
run = Path(sys.argv[1])
info = json.loads((run / "run.json").read_text())
rows = []
for condition in ("v1", "abl", "v2"):
    dest = run / "reproduce_s42" / condition
    if json.loads((dest / "status.json").read_text())["status"] != "complete":
        raise SystemExit(f"Incomplete reproduction: {condition}")
    for test, result in json.loads((dest / "results.json").read_text()).items():
        rows.append({"condition": condition, "seed": 42, "steps": 189 if condition == "v1" else 240, "group": "three_seeds", "source": str(dest.relative_to(run)), **result})
for job in info["jobs"]:
    dest = run / job["name"]
    if json.loads((dest / "status.json").read_text())["status"] != "complete":
        raise SystemExit(f"Incomplete training: {job['name']}")
    for test, result in json.loads((dest / "results.json").read_text()).items():
        rows.append({"condition": job["condition"], "seed": job["seed"], "steps": job["expected_steps"], "group": "step_control" if job["name"] == "v1_240steps_s42" else "three_seeds", "source": job["name"], **result})
aggregates = []
for condition in ("v1", "abl", "v2"):
    tests = ["eval_in_domain_no_summary", "eval_cross_repo_no_summary"]
    if condition == "v2":
        tests += ["eval_in_domain", "eval_cross_repo"]
    for test in tests:
        values = [row for row in rows if row["condition"] == condition and row["test"] == test and row["group"] == "three_seeds"]
        if sorted(row["seed"] for row in values) != [42, 43, 44]:
            raise SystemExit(f"Missing seeds: {condition}/{test}")
        scores = [row["accuracy"] for row in values]
        aggregates.append({"condition": condition, "test": test, "seeds": [42, 43, 44], "mean_accuracy": statistics.mean(scores), "sample_std_accuracy": statistics.stdev(scores)})
(run / "results.json").write_text(json.dumps({"metric": "decision_type_match", "seed42_source": "today_reproduction", "rows": rows, "three_seed_summary": aggregates}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
PHASE=complete
ITEM=all
status "$RUN" complete
printf '%s queue complete; server remains running.\n' "$(date -u +%FT%TZ)"

