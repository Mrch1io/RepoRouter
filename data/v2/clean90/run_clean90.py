import hashlib
import json
import os
import runpy
import shlex
import socket
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
EXP = ROOT / "data/v2/clean90"
SOURCE = ROOT / "data/v2/frozen90/source"
run = json.loads((EXP / "run.json").read_text("utf-8"))
ssh = ["ssh", "-p", "2222", "-o", "BatchMode=yes", "-o", "PasswordAuthentication=no", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]
host = "root@example.invalid"

os.environ.update(Chroma_persist_dir="./data/indices/src_frozen90_clean", bm25_persist_dir="./data/indices/src_frozen90_clean/bm25_index.pkl")
env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
sys.dont_write_bytecode = True
server = tunnel = None
remote_pid = None
argv = None


def save(status):
    run.update(status=status, updated_at=datetime.now(timezone.utc).isoformat())
    (EXP / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(status, flush=True)


def remote(command):
    return subprocess.check_output(ssh + [host, command], text=True, encoding="utf-8", timeout=30).strip()


def check_frozen():
    #1.核对源码和环境文件的 SHA256，保证两组实验使用同一份输入
    for path, digest in run["source_sha256_frozen"].items():
        assert hashlib.sha256((SOURCE / path).read_bytes()).hexdigest() == digest, path
    assert hashlib.sha256((ROOT / ".env").read_bytes()).hexdigest() == run["environment_file_sha256"], ".env SHA256 mismatch"
    index = ROOT / "data/indices/src_frozen90_clean"
    derivation = json.loads((EXP / "index_derivation.json").read_text("utf-8"))
    assert derivation["status"] == "passed"
    assert hashlib.sha256((index / "bm25_index.pkl").read_bytes()).hexdigest() == derivation["index_files_sha256"]["bm25_index.pkl"]
    #2.只读查询表内容，排序后算哈希，避免数据库文件布局或返回顺序影响校验
    with sqlite3.connect((index / "chroma.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        for table, digest in derivation["index_logical_sha256"].items():
            rows = sorted(db.execute(f'SELECT * FROM "{table}"').fetchall(), key=repr)
            assert hashlib.sha256(repr(rows).encode()).hexdigest() == digest, table


def execute(name, args, output):
    #1.每轮先核对输入；输出不能覆盖，日志用 x 模式创建，已有同名文件就报错
    check_frozen()
    assert not output.exists(), f"Refusing overwrite: {output}"
    command = [sys.executable, "-B", *args]
    entry = {"name": name, "argv": command, "started_at": datetime.now(timezone.utc).isoformat(), "exit_code": None}
    run["runs"].append(entry)
    save(name)
    rows = []
    try:
        with (EXP / f"{name}.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        entry["exit_code"] = result.returncode
    finally:
        #2.子进程失败也记录已有输出、失败题号和缺失题号，保留现场方便排查
        entry["ended_at"] = datetime.now(timezone.utc).isoformat()
        entry["invalid_json_lines"] = []
        if output.exists():
            entry["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
            for number, line in enumerate(output.read_text("utf-8").splitlines(), 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        entry["invalid_json_lines"].append(number)
        failures = [r["sample_id"] for r in rows if r.get("execution_error") or r.get("error")]
        entry.update(records=len(rows), failures=failures, missing_ids=sorted({m["id"] for m in run["manifest"]} - {r["sample_id"] for r in rows}))
        save(name + " 已结束，保留本次输出")
    if entry["exit_code"]:
        raise RuntimeError(f"{name}: exit {entry['exit_code']}")
    assert not entry["invalid_json_lines"], entry["invalid_json_lines"]
    assert len(rows) == 90 and {r["sample_id"] for r in rows} == {m["id"] for m in run["manifest"]}
    if failures:
        raise RuntimeError(f"{name}: failed records {failures}; preserve this attempt")


def port_open():
    with socket.socket() as connection:
        connection.settimeout(2)
        return connection.connect_ex(("127.0.0.1", 8080)) == 0


def stop_services():
    global remote_pid
    evidence = {"started_at": datetime.now(timezone.utc).isoformat(), "exit_code": None}
    run["service_shutdown"] = evidence
    errors = []
    #1.只结束日志标记且命令行核对一致的本实验进程
    try:
        service_log = EXP / "router_service.log"
        if remote_pid is None and service_log.exists():
            for line in service_log.read_text("utf-8").splitlines():
                if line.startswith("REPOAGENT_SERVER_PID="):
                    remote_pid = int(line.split("=", 1)[1])
        evidence["remote_pid"] = remote_pid
        if remote_pid is not None:
            current = remote(f"if [ -r /proc/{remote_pid}/cmdline ]; then tr '\\0' ' ' < /proc/{remote_pid}/cmdline; fi")
            evidence["command_before_stop"] = current
            if current:
                assert shlex.split(current) == argv, "Refusing to kill a PID whose command differs"
                remote(f"kill {remote_pid}; for i in $(seq 1 50); do if ! kill -0 {remote_pid} 2>/dev/null; then exit 0; fi; sleep 0.2; done; exit 1")
            evidence["remote_pid_gone"] = remote(f"if [ -e /proc/{remote_pid} ]; then echo present; else echo absent; fi") == "absent"
            assert evidence["remote_pid_gone"], "Experiment PID still exists"
        else:
            evidence["remote_pid_gone"] = server is None
            assert server is None, "Server SSH started but experiment PID was not captured"
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    finally:
        #2.关闭并等待本实验的隧道和 SSH 进程
        for name, process in (("tunnel", tunnel), ("server_ssh", server)):
            if process is not None:
                try:
                    if process.poll() is None:
                        process.terminate()
                    evidence[name + "_exit_code"] = process.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired) as error:
                    errors.append(f"{name}: {error}")
    #3.停止证据独立记录，不能只凭已发出 kill 就算成功
    try:
        evidence["local_8080_connectable"] = port_open()
        assert not evidence["local_8080_connectable"], "Local port 8080 is still reachable"
        if server is not None:
            gpu = remote("nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader")
            evidence["gpu_processes_csv_after_stop"] = gpu
            pids = [int(line.split(",", 1)[0].strip()) for line in gpu.splitlines() if line.strip()]
            evidence["experiment_pid_absent_from_gpu"] = remote_pid is not None and remote_pid not in pids
            assert evidence["experiment_pid_absent_from_gpu"], "Experiment GPU process is not confirmed stopped"
        else:
            evidence["experiment_server_never_started"] = True
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    evidence.update(ended_at=datetime.now(timezone.utc).isoformat(), exit_code=1 if errors else 0, errors=errors)
    run["router_service_stopped"] = not errors and evidence.get("remote_pid_gone", False)
    run["ssh_tunnel_stopped"] = not evidence.get("local_8080_connectable", True) and (tunnel is None or tunnel.poll() is not None)
    save("服务清理失败，请检查证据" if errors else "本实验服务和隧道已停止")
    if errors:
        raise RuntimeError("Service cleanup failed: " + "; ".join(errors))


try:
    try:
        check_frozen()
        output = EXP / "e2e_native.jsonl"
        execute("generate_native", [str(SOURCE / "scripts/eval_agent.py"), "--routing-backend", "native", "--model", "deepseek-flash", "--max-tokens", "4096", "--jsonl-out", str(output)], output)

        run["q4_preflight"] = {"started_at": datetime.now(timezone.utc).isoformat(), "exit_code": None}
        save("native 完成，开始 Q4 预检")
        assert not port_open(), "Local port 8080 is already occupied"
        old = json.loads((ROOT / "data/v2/pilot_full_answers/run.json").read_text("utf-8"))["q4_preflight"]["runtime"]
        argv = old["llama_server"]["argv"]
        assert old["gguf"]["sha256"] == "c3b83fb8e5693e1e8042fbb46a9d9b808674bc008e544562be4d488a6ce02dfd"
        assert old["llama_server"]["commit"] == "a37cc46"
        #flock 协调使用同一把锁的实验；exec 和 --no-fork 保持 PID，方便结束时核对服务进程
        launch = 'set -e; test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; exec ' + shlex.join(argv)
        command = "echo REPOAGENT_SERVER_PID=$$; exec flock --no-fork -n -x /root/autodl-tmp/gpu.lock sh -c " + shlex.quote(launch)
        with (EXP / "router_service.log").open("x", encoding="utf-8") as log:
            server = subprocess.Popen(ssh + [host, command], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        with (EXP / "ssh_tunnel.log").open("x", encoding="utf-8") as log:
            #-N 只做端口转发，-L 把本机 8080 接到远端的 8080 服务，两端都使用回环地址
            tunnel = subprocess.Popen(ssh + ["-o", "ExitOnForwardFailure=yes", "-N", "-L", "127.0.0.1:8080:127.0.0.1:8080", host], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        run["local_processes"] = {"runner": os.getpid(), "server_ssh": server.pid, "tunnel": tunnel.pid}
        save("等待本实验 Q4 服务就绪")
        deadline = time.monotonic() + 180
        while True:
            for line in (EXP / "router_service.log").read_text("utf-8").splitlines():
                if line.startswith("REPOAGENT_SERVER_PID="):
                    remote_pid = int(line.split("=", 1)[1])
            if server.poll() is not None or tunnel.poll() is not None:
                raise RuntimeError("SSH server or tunnel exited; inspect experiment logs")
            if time.monotonic() >= deadline:
                raise TimeoutError("Q4 service did not become ready within 180 seconds")
            try:
                with urlopen("http://127.0.0.1:8080/health", timeout=3) as response:
                    if json.load(response)["status"] == "ok":
                        break
            except (URLError, TimeoutError, ConnectionResetError, RemoteDisconnected):
                pass
            time.sleep(1)
        assert remote_pid is not None, "Port 8080 is not served by this experiment"
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
        template_sha = hashlib.sha256(props["chat_template"].encode()).hexdigest()
        assert template_sha == old["llama_server"]["chat_template_sha256"]
        #执行冻结脚本的顶层初始化，设置源码导入路径；默认名称不是 __main__，不会启动评测
        runpy.run_path(str(SOURCE / "scripts/eval_agent.py"))
        from src.agent.router import route
        started = time.perf_counter()
        question = "请定位当前仓库 src/agent/core.py 中 RepoAgent.chat 的实现。"
        decision = route(question)
        assert decision.get("decision_type") == "tool_call" and decision.get("tool_name") == "search_codebase" and not decision.get("parse_failed"), decision
        run["q4_preflight"].update(ended_at=datetime.now(timezone.utc).isoformat(), exit_code=0, runtime={"gguf": {**old["gguf"], "sha256": model_sha}, "llama_server": {"pid": remote_pid, "argv": argv, "actual_command": actual_command, "version": version, "commit": old["llama_server"]["commit"], "chat_template_sha256": template_sha, "props": props}, "hardware_csv": remote("nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv"), "gpu_processes_csv": remote("nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv"), "smoke_test": {"question": question, "decision": decision, "seconds": time.perf_counter() - started}, "ssh_tunnel": "127.0.0.1:8080 -> 5090 127.0.0.1:8080", "gpu_lock_held": True})
        save("Q4 参数与真实路由验证通过")
        output = EXP / "e2e_q4.jsonl"
        execute("generate_q4", [str(SOURCE / "scripts/eval_agent.py"), "--routing-backend", "phase2a", "--model", "deepseek-flash", "--max-tokens", "4096", "--jsonl-out", str(output)], output)
    finally:
        #预检或评测中途报错也会走到这里，确保尝试关闭本实验启动的服务和隧道
        preflight = run.get("q4_preflight")
        if preflight and preflight.get("exit_code") is None:
            preflight.update(ended_at=datetime.now(timezone.utc).isoformat(), exit_code=1)
        stop_services()
    check_frozen()
    save("两条链路各 90 题生成完成，服务与隧道已停止")
except Exception as error:
    run["error"] = f"{type(error).__name__}: {error}"
    save("实验暂停：保留已有记录，不覆盖重跑")
    raise
