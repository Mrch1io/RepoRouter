"""
轻量级端到端评测：Query → 工具决策 → 检索(+软熔断) → 最终回答，并落盘 JSONL 链路日志。

依赖：.env 已配置 DeepSeek + Qwen 嵌入；已建向量库。

用法（项目根）：
  python scripts/eval_agent.py --limit 5
  python scripts/eval_agent.py --model deepseek-flash --repeat 3          # 默认写到 data/v2/e2e_deepseek-flash.jsonl
  python scripts/eval_agent.py --model deepseek-v4-pro --repeat 3
  python scripts/eval_agent.py --model deepseek-flash --thinking --repeat 3
  python scripts/eval_agent.py --routing-backend phase2a                  # 微调小模型派单（需先启动路由服务）
  python scripts/eval_agent.py --soft-fuse --soft-fuse-threshold 1.12
  python scripts/eval_agent.py --bucket domain --limit 10
  python scripts/eval_agent.py --phase1 --model deepseek-flash          # 固定 22 题试评

新增：每条记录带 cost_usd / seconds / 缓存命中；终端摘要带 95% 置信区间。

指标（终端摘要）：
  第一层 Tool Decision：与 should_call_tool 对照（yes/no/optional）；optional=prefer_ground，调与不调均计为正确。
  子集「仅 must_ground」：应检索条数、已检索数、未检索(FN)、检索率。
  第二层（仅实际发生 search_codebase 的样本）：软熔断触发率、与 should_soft_fallback_if_searched 的对照
  第三层 final_judgement / final_answer_type 默认 pending，供人工或后续 LLM-as-judge 补全
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.core import RepoAgent
from src.config import settings
from src.evaluation.e2e_manifest import build_e2e_manifest
from src.evaluation.llm_judge_e2e import PHASE1_JUDGE_SAMPLE_IDS
from src.evaluation.stats import wilson_ci
from src.agent.tools.tools_schemas import check_tool_args


def _extract_search_from_trace(trace: list | None) -> tuple[bool, float | None, bool | None]:
    """是否调用了 search_codebase；最后一次的 top1_distance、soft_fuse_triggered。"""
    if not trace:
        return False, None, None
    called = False
    top1: float | None = None
    fused: bool | None = None
    for ev in trace:
        if not isinstance(ev, dict):
            continue
        if ev.get("tool_name") == "search_codebase":
            called = True
            sp = ev.get("search_payload")
            if isinstance(sp, dict):
                top1 = sp.get("top1_distance")
                if top1 is not None:
                    top1 = float(top1)
                fused = sp.get("soft_fuse_triggered")
    return called, top1, fused


def _tool_names_from_trace(trace: list | None) -> list[str]:
    names: list[str] = []
    if not trace:
        return names
    for ev in trace:
        if isinstance(ev, dict) and ev.get("event") == "assistant_tool_calls":
            names.extend(ev.get("names") or [])
    return names


def _compare(path_a: Path, path_b: Path) -> None:
    """两份结果按 (题号, 第几次) 配对：只看两边对错不一样的题，用 McNemar 检验差异是不是偶然。"""
    from src.evaluation.stats import mcnemar_p

    def load(p: Path) -> dict:
        rows = [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
        return {(r["sample_id"], r.get("repeat", 0)): r for r in rows}

    a, b = load(path_a), load(path_b)
    keys = sorted(set(a) & set(b))
    only_a = sum(1 for k in keys if a[k]["l1_correct"] and not b[k]["l1_correct"])
    only_b = sum(1 for k in keys if b[k]["l1_correct"] and not a[k]["l1_correct"])
    for name, d in ((path_a.name, a), (path_b.name, b)):
        ok = sum(d[k]["l1_correct"] for k in keys)
        cost = sum(d[k].get("cost_usd") or 0 for k in keys) / max(len(keys), 1)
        sec = sum(d[k].get("seconds") or 0 for k in keys) / max(len(keys), 1)
        print(f"  {name}: 一致 {ok}/{len(keys)} | 每题 ${cost:.5f} | {sec:.1f} 秒")
    print(f"  A对B错 {only_a} 题，A错B对 {only_b} 题，McNemar p = {mcnemar_p(only_a, only_b):.4f}（< 0.05 才算显著差异）")


def main() -> None:
    parser = argparse.ArgumentParser(description="RepoAgent E2E pipeline eval (JSONL trace)")
    parser.add_argument("--limit", type=int, default=0, help="0 = 跑 manifest 全部样本")
    parser.add_argument("--phase1", action="store_true", help="按既定顺序跑 22 题试评，不能与 --limit / --bucket 联用")
    parser.add_argument(
        "--bucket",
        type=str,
        default="all",
        choices=["all", "domain", "far_general", "term_near_no_rag", "misleading_edge"],
    )
    #默认输出改到 data/v2/，不会覆盖 3 月的 data/e2e_eval_log*.jsonl
    parser.add_argument("--jsonl-out", type=Path, default=None)
    parser.add_argument("--soft-fuse", action="store_true", help="启用 config 软熔断（需阈值）")
    parser.add_argument("--soft-fuse-threshold", type=float, default=1.12)
    parser.add_argument("--model", default=settings.model_name, help="deepseek-flash / deepseek-v4-pro")
    parser.add_argument("--max-tokens", type=int, default=800, help="每次回答调用的输出 token 上限")
    parser.add_argument("--thinking", action="store_true", help="打开 DeepSeek 思考模式")
    parser.add_argument("--routing-backend", default=settings.routing_backend, choices=["native", "phase2a"])
    parser.add_argument("--repeat", type=int, default=1, help="同一批题重复跑几次（大模型有随机性）")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("A.jsonl", "B.jsonl"), help="不跑模型，只对比两份结果（McNemar）")
    args = parser.parse_args()
    if args.compare:
        _compare(*args.compare)
        return
    if args.repeat < 1 or args.max_tokens < 1 or args.limit < 0:
        parser.error("--repeat / --max-tokens 必须大于 0，--limit 不能为负数")
    if args.phase1 and (args.limit or args.bucket != "all"):
        parser.error("--phase1 不能与 --limit / --bucket 联用")
    if args.jsonl_out is None:
        tag = "phase2a" if args.routing_backend == "phase2a" else args.model + ("-thinking" if args.thinking else "")
        if args.phase1:
            tag += "_phase1"
        args.jsonl_out = ROOT / "data" / "v2" / f"e2e_{tag}.jsonl"

    manifest = build_e2e_manifest()
    if args.phase1:
        by_id = {row["id"]: row for row in manifest}
        manifest = [by_id[sample_id] for sample_id in PHASE1_JUDGE_SAMPLE_IDS]
    if args.bucket != "all":
        manifest = [m for m in manifest if m["bucket_type"] == args.bucket]
    if args.limit > 0:
        manifest = manifest[: args.limit]

    jsonl_path = args.jsonl_out
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    from unittest.mock import patch as mock_patch

    stack = ExitStack()
    if args.soft_fuse:
        stack.enter_context(mock_patch.object(settings, "soft_fuse_enabled", True))
        stack.enter_context(mock_patch.object(settings, "soft_fuse_distance_threshold", args.soft_fuse_threshold))
    else:
        stack.enter_context(mock_patch.object(settings, "soft_fuse_enabled", False))
        stack.enter_context(mock_patch.object(settings, "soft_fuse_distance_threshold", None))
    stack.enter_context(mock_patch.object(settings, "model_name", args.model))
    stack.enter_context(mock_patch.object(settings, "llm_thinking", args.thinking))
    stack.enter_context(mock_patch.object(settings, "routing_backend", args.routing_backend))

    #每次重复各自的“一致”条数，以及花费 / 耗时 / 缓存
    per_rep_correct = [0] * args.repeat
    costs: list[float] = []
    secs: list[float] = []
    cache_hit = cache_miss = 0
    no_n = 0                  #不该调工具的题数（BFCL 无关检测的分母）
    no_correct = 0
    execution_errors = 0
    args_ok = args_total = 0  #实际发生的工具调用里，参数合法的有几次

    l1_correct = 0
    l1_fp = 0
    l1_fn = 0
    must_n = 0
    must_ok = 0
    must_fn = 0
    l2_searched = 0
    l2_fuse_expected_and_triggered = 0
    l2_fuse_expected = 0

    with stack:
        with jsonl_path.open("w", encoding="utf-8") as fp_out:
            #同一批题按顺序重复跑 repeat 次
            for completed, (rep, row) in enumerate(((r, m) for r in range(args.repeat) for m in manifest), 1):
                q = row["query"]
                trace: list = []
                final_text = ""
                execution_error = None
                started = perf_counter()
                try:
                    agent = RepoAgent()
                    agent.clear_memory()
                    final_text = agent.chat(q, trace=trace, max_tokens=args.max_tokens)
                except Exception as e:
                    execution_error = f"{type(e).__name__}: {e}"
                elapsed = round(perf_counter() - started, 2)
                finish_reason = next((ev["finish_reason"] for ev in reversed(trace) if ev.get("event") == "answer_metadata"), None)
                if finish_reason == "iteration_limit":
                    execution_error = "IterationLimit: 工具调用轮数达到上限，未生成最终回答"
                tool_errors = [f"{ev['tool_name']}: {ev['error']}" for ev in trace if "tool_name" in ev and ev.get("error")]
                if execution_error is None and tool_errors:
                    execution_error = "; ".join(tool_errors)
                if execution_error is None and not (final_text or "").strip():
                    execution_error = "EmptyAnswer: 模型未返回回答内容"
                execution_errors += execution_error is not None

                actual_call, top1_dist, soft_fb = _extract_search_from_trace(trace)
                expect = row["should_call_tool"]
                ok_l1 = execution_error is None and (expect == "optional" or (expect == "yes") == actual_call)
                l1_correct += ok_l1
                l1_fn += execution_error is None and expect == "yes" and not actual_call
                l1_fp += execution_error is None and expect == "no" and actual_call
                no_n += expect == "no"
                no_correct += expect == "no" and ok_l1
                #每次工具调用都按 schema 查一遍参数（借鉴 BFCL 的 AST 检查）
                for ev in trace:
                    if "tool_name" in ev and "arguments" in ev:
                        args_total += 1
                        args_ok += check_tool_args(ev["tool_name"], ev.get("arguments"))
                per_rep_correct[rep] += ok_l1

                #本题的花费、耗时、缓存命中（core.chat 写在 trace 最后的 usage 事件里）
                usage = next((ev for ev in trace if ev.get("event") == "usage"), {})
                costs.append(usage.get("cost_usd", 0.0))
                secs.append(elapsed)
                cache_hit += usage.get("cache_hit", 0)
                cache_miss += usage.get("cache_miss", 0)

                gt = row.get("grounding_tier")
                if gt == "must_ground":
                    must_n += 1
                    if execution_error is None and actual_call:
                        must_ok += 1
                    elif execution_error is None:
                        must_fn += 1

                if actual_call:
                    l2_searched += 1
                    if row["should_soft_fallback_if_searched"] == "yes":
                        l2_fuse_expected += 1
                        if soft_fb is True:
                            l2_fuse_expected_and_triggered += 1

                record = {
                    "question": q,
                    "sample_id": row["id"],
                    "repeat": rep,
                    "model": args.model,
                    "max_tokens": args.max_tokens,
                    "thinking": args.thinking,
                    "routing_backend": args.routing_backend,
                    "l1_correct": ok_l1,
                    "cost_usd": usage.get("cost_usd"),
                    "seconds": elapsed,
                    "router_decision": next((ev["decision"] for ev in trace if ev.get("event") == "router_decision"), None),
                    "bucket_type": row["bucket_type"],
                    "grounding_tier": row.get("grounding_tier"),
                    "should_call_tool": row["should_call_tool"],
                    "actual_call_search_codebase": actual_call,
                    "tool_names_round": _tool_names_from_trace(trace),
                    "should_hit_private_knowledge": row["should_hit_private_knowledge"],
                    "should_soft_fallback_if_searched": row["should_soft_fallback_if_searched"],
                    "top1_distance": top1_dist,
                    "soft_fallback_triggered": soft_fb,
                    "execution_error": execution_error,
                    "final_answer": final_text or "",
                    "final_answer_preview": (final_text or "")[:1200],
                    "answer_finish_reason": finish_reason,
                    "tool_contexts": [
                        {"tool_name": ev["tool_name"], "content": ev["content"]}
                        for ev in trace if ev.get("event") == "tool_context"
                    ],
                    "final_answer_type": "pending",
                    "final_judgement": "pending",
                    "trace_events": len(trace),
                    "soft_fuse_config": {
                        "enabled": args.soft_fuse,
                        "threshold": args.soft_fuse_threshold if args.soft_fuse else None,
                    },
                }
                fp_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                fp_out.flush()
                status = execution_error or f"已保存 {len(final_text or '')} 字符"
                print(f"[{completed}/{len(manifest) * args.repeat}] {row['id']}: {status}", flush=True)

    n = len(manifest) * args.repeat
    lo, hi = wilson_ci(l1_correct, n)
    print(f"\n配置：model={args.model} thinking={args.thinking} routing={args.routing_backend} repeat={args.repeat}")
    print("\n=== 第一层：Tool Decision（search_codebase）===")
    print(f"  样本数: {n}（{len(manifest)} 题 × {args.repeat} 次）")
    print(f"  运行失败: {execution_errors}（单独记录，不计入正确数或 FP/FN）")
    print(f"  决策与预期一致: {l1_correct}/{n}  ({(l1_correct/n*100 if n else 0):.1f}%)  95%CI [{lo*100:.1f}, {hi*100:.1f}]")
    if args.repeat > 1:
        print(f"  每次分别: {per_rep_correct}（最少 {min(per_rep_correct)}，最多 {max(per_rep_correct)}）")
    print(f"  False Positive（不该调却调）: {l1_fp}")
    print(f"  False Negative（该调未调）: {l1_fn}")
    print("  —— 子集 grounding_tier=must_ground（应检索）——")
    print(f"  must_ground 条数: {must_n}  |  已检索: {must_ok}  |  未检索(FN): {must_fn}")
    if must_n:
        print(f"  must_ground 检索率: {100.0 * must_ok / must_n:.1f}%")

    #借鉴 BFCL：把“该调时调了”和“不该调时没调”分开报，比一个混在一起的一致率更清楚
    print("  —— BFCL 口径 ——")
    for name, k, m in (("相关检测（该查库时查了）", must_ok, must_n), ("无关检测（不该查库时没查）", no_correct, no_n)):
        if m:
            lo2, hi2 = wilson_ci(k, m)
            print(f"  {name}: {k}/{m} ({100.0 * k / m:.1f}%)  95%CI [{lo2*100:.1f}, {hi2*100:.1f}]")
    if args_total:
        print(f"  参数合法率（必填参数齐、类型对、没编参数）: {args_ok}/{args_total} ({100.0 * args_ok / args_total:.1f}%)")

    print("\n=== 第二层：检索 + 软熔断（仅 actual_call_search_codebase 的样本）===")
    print(f"  实际触发检索次数: {l2_searched}")
    print(f"  其中「预期应软熔断」的样本数: {l2_fuse_expected}")
    if l2_fuse_expected:
        print(
            f"  预期应熔断且已熔断: {l2_fuse_expected_and_triggered}/{l2_fuse_expected}  "
            f"({100.0 * l2_fuse_expected_and_triggered / l2_fuse_expected:.1f}%)"
        )
    else:
        print("  预期应熔断且已熔断: N/A（无「误调检索且预期熔断」样本）")

    if costs:
        print("\n=== 成本与耗时（每题平均）===")
        print(f"  花费: ${sum(costs)/len(costs):.5f}  |  耗时: {sum(secs)/len(secs):.1f} 秒  |  总花费: ${sum(costs):.4f}")
        if cache_hit + cache_miss:
            print(f"  输入 token 缓存命中率: {cache_hit / (cache_hit + cache_miss) * 100:.1f}%")

    print("\n=== 第三层：最终回答 ===")
    print("  final_judgement / final_answer_type 默认为 pending，请对 JSONL 人工标注或接 LLM-as-judge。")

    print(f"\n已写入 {jsonl_path}")


if __name__ == "__main__":
    main()
