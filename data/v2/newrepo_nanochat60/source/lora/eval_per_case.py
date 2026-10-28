import argparse
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys


sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
ROOT = SOURCE.parent
PROJECT = ROOT.parents[2]
FROZEN_DATA = {"questions_frozen.jsonl", "repo_summary.txt",
               "eval_nanochat60.json", "eval_nanochat60_no_summary.json"}


def now()->str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def verify_frozen():
    #1.文件集合也核对，不能漏掉新加的源码
    manifest = ROOT / "frozen_sha256.json"
    raw = manifest.read_bytes()
    hashes = json.loads(raw)["files"]
    actual = FROZEN_DATA | {p.relative_to(ROOT).as_posix()
                            for p in SOURCE.rglob("*") if p.is_file()}
    if set(hashes) != actual:
        raise SystemExit("冻结文件集合不一致，停止评测")
    #2.逐个核对字节，校验完成后才能导入评测代码
    for name, expected in hashes.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise SystemExit(f"冻结文件不存在或越界：{name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"SHA256 不一致：{name}")
    return hashlib.sha256(raw).hexdigest()


def import_evaluator(mode : str):
    #1.保留虚拟环境，排除原项目源码的导入路径
    sys.path[:] = [str(SOURCE)] + [p for p in sys.path
        if not Path(p).resolve().is_relative_to(PROJECT)
        or Path(p).resolve().is_relative_to(PROJECT / ".venv")]
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in {"src", "lora"}:
            file = getattr(module, "__file__", None)
            if file and not Path(file).resolve().is_relative_to(SOURCE):
                raise SystemExit(f"已导入原项目模块，停止评测：{name}")
    return importlib.import_module(f"lora.eval_routing_{mode}")


def load_generator(args, evaluator, stack):
    #1.API 参数和原脚本一致，只初始化一次客户端
    if args.mode == "api":
        client, default_model = evaluator._openai_client_and_model()
        stack.callback(client.close)
        args.model = args.model or default_model
        extra = ({"thinking": {"type": "disabled"}} if args.model.startswith("deepseek")
                 else {"chat_template_kwargs": {"enable_thinking": False}})

        def generate(case):
            messages = [{"role": role, "content": case["conversations"][i]["value"]}
                        for i, role in enumerate(("system", "user"))]
            response = client.chat.completions.create(
                model=args.model, temperature=args.temperature,
                max_tokens=args.max_tokens, messages=messages, extra_body=extra)
            return (response.choices[0].message.content or "").strip()

        return generate
    #2.本地模型沿用原脚本的加载、LoRA 检查和生成函数
    from transformers import AutoTokenizer

    args.base_model = evaluator._resolve_pretrained_source(
        args.base_model or evaluator.DEFAULT_BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = evaluator._load_causal_lm(args.base_model)
    if args.mode == "lora":
        from peft import PeftModel

        args.lora_path = evaluator._resolve_pretrained_source(
            args.lora_path or evaluator.DEFAULT_LORA_PATH)
        model = PeftModel.from_pretrained(model, args.lora_path)
        lora_b = [p for name, p in model.named_parameters() if "lora_B" in name]
        if not lora_b or all(float(p.abs().sum()) == 0 for p in lora_b):
            raise SystemExit("LoRA 权重没有加载进来（lora_B 全为 0）")
        print(f"LoRA 已加载：{len(lora_b)} 个 lora_B 矩阵")
    model.eval()

    def generate(case):
        return evaluator.transformers_generate_reply(
            tokenizer, model, case, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, do_sample=False)

    return generate


def output_paths(prefix):
    return [Path(str(prefix) + suffix) for suffix in (".log", ".jsonl", ".metrics.json")]


def evaluate(args, evaluator, generate, data, prefix, started, manifest_hash):
    #1.每份测试集单独计分，文件只允许新建
    cases = json.loads(data.read_text(encoding="utf-8"))
    metrics = evaluator.Phase2aMetrics(n_file_rows=len(cases))
    buckets = {name: {"correct": 0, "total": 0} for name in "ABCDE"}
    pairs = {name: [] for name in ("bpe", "muon", "kv_cache", "gsm8k")}
    failures = 0
    prefix.parent.mkdir(parents=True, exist_ok=True)
    log_path, rows_path, metrics_path = output_paths(prefix)
    with (log_path.open("x", encoding="utf-8") as log,
          rows_path.open("x", encoding="utf-8") as rows,
          metrics_path.open("x", encoding="utf-8") as stats):

        def emit(line):
            print(line, flush=True)
            print(line, file=log, flush=True)

        emit(f"开始：{started}，模式：{args.mode}，测试集：{data.name}")
        emit(f"冻结文件 SHA256 校验通过，清单 SHA256：{manifest_hash}")
        #2.逐题正确性取原评分器的增量，不重写评分条件
        for i, case in enumerate(cases):
            gold = evaluator.gold_from_sharegpt_case(case)
            raw, pred, error = "", None, None
            decision_match, tool_match = False, None
            if gold is None:
                metrics.skipped_bad_gold += 1
            else:
                try:
                    raw = generate(case)
                except Exception as exc:
                    if args.mode != "api":
                        raise
                    failures += 1
                    status = getattr(exc, "status_code", None)
                    error = {"type": type(exc).__name__,
                             "status_code": status if isinstance(status, int) else None}
                    emit(f"[warn] 第 {i + 1} 条 API 失败：{error}")
                pred = evaluator.parse_routing_json(raw) if raw else None
                before_decision, before_tool = metrics.n_decision_match, metrics.n_tool_match
                metrics.update(gold, pred)
                decision_match = metrics.n_decision_match > before_decision
                if gold["decision_type"] == "tool_call":
                    tool_match = metrics.n_tool_match > before_tool
            record = {"id": case["id"], "bucket": case["bucket"],
                      "pair_id": case["pair_id"],
                      "gold_decision_type": gold["decision_type"] if gold else None,
                      "raw_output": raw, "parsed_output": pred,
                      "decision_match": decision_match, "tool_match": tool_match,
                      "api_error": error}
            rows.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows.flush()
            bucket = buckets[case["bucket"]]
            bucket["total"] += 1
            bucket["correct"] += decision_match
            if case["pair_id"] is not None:
                pairs[case["pair_id"]].append(decision_match)
            emit(f"进度：{i + 1}/{len(cases)}")
        #3.汇总文字直接用原脚本，桶和配对组只聚合逐题结果
        report = metrics.report_lines()
        for line in report:
            emit(line)
        ended = now()
        pair_results = {name: {"correct": sum(values), "total": len(values),
                               "all_correct": bool(values) and all(values)}
                        for name, values in pairs.items()}
        result = {"started_at": started, "ended_at": ended,
                  "executable": sys.executable, "argv": sys.argv,
                  "test_data": str(data), "frozen_manifest_sha256": manifest_hash,
                  "parameters": {name: getattr(args, name) for name in
                      ("mode", "model", "base_model", "lora_path", "temperature",
                       "max_tokens", "max_new_tokens")},
                  "thinking": False, "do_sample": False if args.mode != "api" else None,
                  "api_failures": failures, "metrics": vars(metrics),
                  "buckets": buckets, "pairs": pair_results,
                  "pairs_all_correct": sum(p["all_correct"] for p in pair_results.values()),
                  "report_lines": report}
        json.dump(result, stats, ensure_ascii=False, indent=2)
        stats.write("\n")
        emit(f"结束：{ended}")


def main():
    #1.先检查全部输出目标，避免已测一半才发现会覆盖历史结果
    parser = argparse.ArgumentParser(description="冻结测试集逐题路由评测")
    parser.add_argument("--mode", choices=("base", "lora", "api"), required=True)
    parser.add_argument("--test-data", type=Path, nargs="+", required=True)
    parser.add_argument("--output-prefix", type=Path, nargs="+", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if len(args.test_data) != len(args.output_prefix):
        parser.error("--test-data 与 --output-prefix 必须一一对应")
    data_paths = [p.resolve() for p in args.test_data]
    prefixes = [p.resolve() for p in args.output_prefix]
    allowed = {ROOT / name for name in FROZEN_DATA if name.startswith("eval_")}
    if any(p not in allowed for p in data_paths):
        parser.error("测试文件必须是实验目录内的两份冻结评测文件")
    outputs = [p for prefix in prefixes for p in output_paths(prefix)]
    if any(not p.is_relative_to(ROOT / "results") for p in outputs):
        parser.error("输出必须位于实验目录的 results/ 内")
    if len(set(outputs)) != len(outputs) or any(p.exists() for p in outputs):
        parser.error("输出重复或文件已存在，停止评测")
    #2.每份文件评测前重新校验；模型只加载一次
    evaluator, generate = None, None
    with ExitStack() as stack:
        for data, prefix in zip(data_paths, prefixes):
            started = now()
            manifest_hash = verify_frozen()
            if evaluator is None:
                evaluator = import_evaluator(args.mode)
                generate = load_generator(args, evaluator, stack)
            evaluate(args, evaluator, generate, data, prefix, started, manifest_hash)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        #配置异常可能带输入值，这里只输出类型，避免泄露密钥
        raise SystemExit(f"评测中止：{type(exc).__name__}") from None
