#!/usr/bin/env bash
set -euo pipefail

RUN=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=/root/autodl-tmp/devagent
PY="$ROOT/venv/bin/python"
BASE=/root/autodl-tmp/models/Qwen/Qwen3.5-4B
LOCK=/root/autodl-tmp/gpu.lock

status() {
    printf '{"stage":"%s","updated_at_utc":"%s"}\n' "$1" "$(date -u +%FT%TZ)" > "$RUN/status.json"
}

exec 9<>"$LOCK"
flock -n 9
[[ ! -s "$LOCK" ]] || { printf 'GPU lock contains an existing owner.\n' >&2; exit 1; }
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z "$apps" ]] || { printf 'GPU is busy: %s\n' "$apps" >&2; exit 1; }
[[ ! -e "$RUN/adapter" ]] || { printf 'Adapter directory already exists.\n' >&2; exit 1; }

finish() {
    code=$?
    trap - EXIT
    if (( code != 0 )); then status failed; fi
    : > "$LOCK"
    exit "$code"
}
trap finish EXIT
printf 'RepoAgent v2 no-summary ablation, runner PID %s, %s\n' "$$" "$RUN" > "$LOCK"
printf '%s\n' "$$" > "$RUN/runner.pid"
cd "$RUN"
sha256sum --check inputs.sha256
export PATH="$ROOT/venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

status training
cd "$ROOT/LLaMA-Factory"
"$ROOT/venv/bin/llamafactory-cli" train "$RUN/train.yaml" > "$RUN/train.log" 2>&1
"$PY" - "$RUN/adapter" <<'PY'
import json
import sys
from pathlib import Path

adapter = Path(sys.argv[1])
results = json.loads((adapter / "train_results.json").read_text())
state = json.loads((adapter / "trainer_state.json").read_text())
if results["epoch"] != 3.0 or state["global_step"] != 240:
    raise SystemExit("Training did not complete all 240 steps / 3 epochs.")
if not (adapter / "adapter_model.safetensors").is_file():
    raise SystemExit("Trained adapter is missing.")
PY

status evaluating
cd "$RUN/source"
for test in eval_in_domain_no_summary eval_cross_repo_no_summary; do
    "$PY" loRA/eval_routing_lora.py \
        --base-model "$BASE" --lora-path "$RUN/adapter" \
        --test-data "$RUN/data/$test.json" \
        --max-new-tokens 512 --temperature 0.1 > "$RUN/$test.log" 2>&1
done
status complete
