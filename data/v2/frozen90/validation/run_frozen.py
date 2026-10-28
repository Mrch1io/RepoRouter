import hashlib
import json
import os
import runpy
import shlex
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from http.client import RemoteDisconnected
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path.cwd()
EXP = ROOT / "data/v2/frozen90"
SOURCE = EXP / "source"
LOG_SUFFIX = "_attempt3"
run = json.loads((EXP / "run.json").read_text("utf-8"))
ssh = ["ssh", "-p", "49616", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]
host = "root@example.invalid"
env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
server = tunnel = None
remote_pid = None


def save(status):
    run.update(status=status, updated_at=datetime.now(timezone.utc).isoformat())
    (EXP / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(status, flush=True)


def remote(command):
    return subprocess.check_output(ssh + [host, command], text=True, encoding="utf-8", timeout=30).strip()


def check_frozen():
    for path, digest in run["source_sha256_frozen"].items():
        assert hashlib.sha256((SOURCE / path).read_bytes()).hexdigest() == digest, path
    assert hashlib.sha256((EXP / "review_plan.json").read_bytes()).hexdigest() == run["review_plan_sha256"]
    assert hashlib.sha256((ROOT / ".env").read_bytes()).hexdigest() == run["environment_file_sha256"]
    index = ROOT / "data/indices/src"
    assert hashlib.sha256((index / "bm25_index.pkl").read_bytes()).hexdigest() == run["index_files_sha256_before"]["bm25_index.pkl"]
    with sqlite3.connect((index / "chroma.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        for table, digest in run["index_logical_sha256_before"].items():
            rows = sorted(db.execute(f'SELECT * FROM "{table}"').fetchall(), key=repr)
            assert hashlib.sha256(repr(rows).encode()).hexdigest() == digest, table


def execute(name, args, output):
    check_frozen()
    assert not output.exists(), f"Refusing overwrite: {output}"
    command = [sys.executable, "-B", *args]
    entry = {"name": name, "argv": command, "started_at": datetime.now(timezone.utc).isoformat()}
    run["runs"].append(entry)
    save(name)
    with (EXP / f"{name}.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    entry.update(exit_code=result.returncode, ended_at=datetime.now(timezone.utc).isoformat())
    if result.returncode:
        raise RuntimeError(f"{name}: exit {result.returncode}")
    entry["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    if output.suffix == ".jsonl":
        rows = [json.loads(line) for line in output.read_text("utf-8").splitlines() if line.strip()]
        assert len(rows) == 90 and {r["sample_id"] for r in rows} == {m["id"] for m in run["manifest"]}
        failures = [r["sample_id"] for r in rows if r.get("execution_error") or r.get("error")]
        entry.update(records=len(rows), failures=failures)
        if failures and name.startswith("generate_"):
            raise RuntimeError(f"{name}: failed records {failures}; preserve this attempt")


try:
    check_frozen()
    old = json.loads((ROOT / "data/v2/pilot_full_answers/run.json").read_text("utf-8"))["q4_preflight"]["runtime"]
    argv = old["llama_server"]["argv"]
    gate = '''from pathlib import Path
import json
p=Path('/root/autodl-tmp/devagent/ablations/v2_no_summary_2026-10-24')
s=json.loads((p/'status.json').read_text())
t=json.loads((p/'adapter/trainer_state.json').read_text())
r=json.loads((p/'adapter/train_results.json').read_text())
assert s['stage']=='complete' and t['global_step']==240 and r['epoch']==3.0
assert (p/'adapter/adapter_model.safetensors').is_file()
for name,n in [('in_domain',50),('cross_repo',87)]:
 text=(p/f'eval_{name}_no_summary.log').read_text()
 assert '====== Phase 2A 路由测评 ======' in text
 assert f'样本总数（文件条数）: {n}' in text and f'金标可解析条数: {n}（跳过金标损坏: 0）' in text
print('ABLATION_GATE='+json.dumps({'stage':s['stage'],'global_step':t['global_step'],'epoch':r['epoch'],'eval_counts':[50,87]}))
'''
    launch = "set -e; /root/miniconda3/bin/python -c " + shlex.quote(gate) + '; test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; exec ' + shlex.join(argv)
    command = "echo REPOAGENT_SERVER_PID=$$; exec flock --no-fork -x /root/autodl-tmp/gpu.lock sh -c " + shlex.quote(launch)
    with (EXP / f"router_service{LOG_SUFFIX}.log").open("x", encoding="utf-8") as log:
        server = subprocess.Popen(ssh + [host, command], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    with (EXP / f"ssh_tunnel{LOG_SUFFIX}.log").open("x", encoding="utf-8") as log:
        tunnel = subprocess.Popen(ssh + ["-o", "ExitOnForwardFailure=yes", "-N", "-L", "127.0.0.1:8080:127.0.0.1:8080", host], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    run["local_processes"] = {"runner": os.getpid(), "server_ssh": server.pid, "tunnel": tunnel.pid}
    save("等待其他实验释放 GPU 锁")
    while True:
        for line in (EXP / f"router_service{LOG_SUFFIX}.log").read_text("utf-8").splitlines():
            if line.startswith("REPOAGENT_SERVER_PID="):
                remote_pid = int(line.split("=", 1)[1])
            if line.startswith("ABLATION_GATE="):
                run["ablation_completion_gate"] = json.loads(line.split("=", 1)[1])
        if server.poll() is not None or tunnel.poll() is not None:
            raise RuntimeError("SSH server or tunnel exited; inspect experiment logs")
        if "ablation_completion_gate" not in run:
            time.sleep(30)
            continue
        try:
            with urlopen("http://127.0.0.1:8080/health", timeout=3) as response:
                if json.load(response)["status"] == "ok":
                    break
        except (URLError, TimeoutError, ConnectionResetError, RemoteDisconnected):
            pass
        time.sleep(30)
    assert remote_pid is not None, "Port 8080 is not served by this experiment"
    assert "ablation_completion_gate" in run
    actual_command = remote(f"tr '\\0' ' ' < /proc/{remote_pid}/cmdline")
    assert shlex.split(actual_command) == argv, actual_command
    model_sha = remote("sha256sum " + shlex.quote(old["gguf"]["path"])).split()[0]
    assert model_sha == old["gguf"]["sha256"]
    version = remote(shlex.quote(argv[0]) + " --version 2>&1")
    assert old["llama_server"]["commit"] in version
    with urlopen("http://127.0.0.1:8080/props", timeout=5) as response:
        props = json.load(response)
    with urlopen("http://127.0.0.1:8080/v1/models", timeout=5) as response:
        models = json.load(response)
    assert models["data"][0]["id"] == "repoagent-router"
    assert hashlib.sha256(props["chat_template"].encode()).hexdigest() == old["llama_server"]["chat_template_sha256"]
    runpy.run_path(str(SOURCE / "scripts/eval_agent.py"))
    from src.agent.router import route
    started = time.perf_counter()
    question = "请定位当前仓库 src/agent/core.py 中 RepoAgent.chat 的实现。"
    decision = route(question)
    assert decision.get("decision_type") == "tool_call" and decision.get("tool_name") == "search_codebase" and not decision.get("parse_failed"), decision
    run["q4_preflight"] = {"runtime": {"gguf": {**old["gguf"], "sha256": model_sha}, "llama_server": {"pid": remote_pid, "argv": argv, "actual_command": actual_command, "version": version, "props": props}, "hardware_csv": remote("nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv"), "gpu_processes_csv": remote("nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv"), "smoke_test": {"question": question, "decision": decision, "seconds": time.perf_counter() - started}, "ssh_tunnel": "127.0.0.1:8080 -> 5090 127.0.0.1:8080", "gpu_lock_held": True}}
    save("隧道与真实路由验证通过，开始相邻运行两条链路")
    for backend, name in (("native", "native"), ("phase2a", "q4")):
        output = EXP / f"e2e_{name}.jsonl"
        execute(f"generate_{name}", [str(SOURCE / "scripts/eval_agent.py"), "--routing-backend", backend, "--model", "deepseek-flash", "--max-tokens", "4096", "--jsonl-out", str(output)], output)
    for name in ("native", "q4"):
        output = EXP / f"judge_{name}.jsonl"
        execute(f"judge_{name}", [str(SOURCE / "scripts/judge_answers.py"), "--jsonl", str(EXP / f"e2e_{name}.jsonl"), "--full", "--out", str(output), "--judge-model", "qwen3.8-max", "--max-tokens", "1200", "--temperature", "0.1", "--workers", "4"], output)
    output = EXP / "comparison.json"
    execute("compare", [str(SOURCE / "scripts/eval_agent.py"), "--compare", str(EXP / "e2e_native.jsonl"), str(EXP / "e2e_q4.jsonl"), "--judges", str(EXP / "judge_native.jsonl"), str(EXP / "judge_q4.jsonl"), "--review-plan", str(EXP / "review_plan.json"), "--compare-out", str(output)], output)
    check_frozen()
    save("两组90题及评分、配对比较完成，待最终校验与打包")
except Exception as error:
    run["error"] = f"{type(error).__name__}: {error}"
    save("实验暂停：保留已有记录，不覆盖重跑")
    raise
finally:
    try:
        if remote_pid is not None and server is not None and server.poll() is None:
            remote(f"kill {remote_pid}")
            run["router_service_stopped"] = True
    finally:
        for process in (tunnel, server):
            if process is not None and process.poll() is None:
                process.terminate()
        run["ssh_tunnel_stopped"] = True
        (EXP / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", "utf-8")
