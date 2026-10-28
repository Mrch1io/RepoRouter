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
import hashlib
import json
import random
import sys
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.core import RepoRouter
from src.config import settings
from src.evaluation.e2e_manifest import build_e2e_manifest
from src.evaluation.llm_judge_e2e import PHASE1_JUDGE_SAMPLE_IDS, load_jsonl_records
from src.evaluation.stats import wilson_ci
from src.agent.tools.schemas import check_tool_args


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


def _retrieval_metrics(row: dict) -> dict:
    called = row["actual_call_search_codebase"]
    result = {"status": "not_searched", "top1_distance": None, "evidence_chars": 0,
              "results_count": 0, "calls": [], "non_payload_messages": 0}
    if not called:
        return result
    contexts = [c for c in row.get("tool_contexts", []) if c["tool_name"] == "search_codebase"]
    arguments = row.get("search_arguments", [])
    next_argument = 0
    for context in contexts:
        try:
            payload = json.loads(context["content"])
        except json.JSONDecodeError:
            result["non_payload_messages"] += 1
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            result["non_payload_messages"] += 1
            continue
        argument_index = next((i for i in range(next_argument, len(arguments)) if arguments[i].get("query") == payload.get("query")), None)
        top_k = None
        if argument_index is not None:
            top_k = arguments[argument_index].get("top_k", 6)
            next_argument = argument_index + 1
        snippets = payload["results"]
        chars = sum(len(item["content"]) for item in snippets)
        result.update(status="searched", top1_distance=payload.get("top1_distance"))
        result["evidence_chars"] += chars
        result["results_count"] += len(snippets)
        result["calls"].append({
            "query": payload.get("query"),
            "top_k": top_k,
            "top1_distance": payload.get("top1_distance"),
            "evidence_chars": chars, "results_count": len(snippets),
        })
    if result["status"] != "searched":
        result.update(status="invalid_context" if contexts else "missing_context", evidence_chars=None, results_count=None)
    return result


def _compare(path_a: Path, path_b: Path, judges=None, out: Path | None = None, review_plan: Path | None = None) -> dict:
    from src.evaluation.stats import bootstrap_ci, mcnemar_p

    a, b = load_jsonl_records(path_a), load_jsonl_records(path_b)
    if a.keys() != b.keys():
        raise ValueError(f"回答未完整配对：仅 A {sorted(a.keys() - b.keys())}；仅 B {sorted(b.keys() - a.keys())}")
    if judges and len(judges) != 2:
        raise ValueError("比较回答质量时必须提供两份评分文件")
    scores = [load_jsonl_records(path) for path in judges] if judges else [{}, {}]
    protocol = None
    protocol_fields = ("judge_gold_mode", "judge_model", "judge_prompt_version", "judge_temperature", "judge_max_tokens")
    for path, rows, answers, answer_rows in zip(judges or [], scores, (path_a, path_b), (a, b)):
        if rows.keys() != a.keys():
            raise ValueError(f"{path}: 评分与回答题号/重复次序不一致")
        digest = hashlib.sha256(answers.read_bytes()).hexdigest()
        for key, row in rows.items():
            if row.get("input_sha256") and row["input_sha256"] != digest:
                raise ValueError(f"{path}: {key} 对应的回答文件校验和不符")
            if "error" in row:
                continue
            #1.完整回答成绩不能混入历史片段，标记还要与实际保存的正文一致
            answer = answer_rows[key].get("final_answer")
            if row.get("answer_format") != "full" or not isinstance(answer,str) or not answer.strip():
                raise ValueError(f"{path}: {key} 评分必须基于完整回答")
            #2.两边和各轮都用同一套评委设置，缺字段时不能假定口径相同
            current = tuple(row.get(field) for field in protocol_fields)
            if (current[0] not in ("legacy","index_aware") or any(value is None for value in current)
                    or any(not isinstance(value,str) or not value.strip() for value in current[:3])):
                raise ValueError(f"{path}: {key} 评分口径字段缺失或无效")
            if protocol is not None and current != protocol:
                raise ValueError(f"{path}: {key} 评分口径不一致：{protocol_fields}")
            protocol = current
            if row.get("final_judgement") not in {"correct", "partially_correct", "incorrect"}:
                raise ValueError(f"{path}: {key} 判定无效")
            if type(row.get("score")) not in (int, float) or not 1 <= row["score"] <= 5:
                raise ValueError(f"{path}: {key} 分数无效")

    pairs = []
    usage_fields = ("calls", "cache_hit", "cache_miss", "output")
    for key in sorted(a):
        pair = {"sample_id": key[0], "repeat": key[1], "grounding_tier": a[key].get("grounding_tier")}
        if pair["grounding_tier"] != b[key].get("grounding_tier") or a[key].get("question") != b[key].get("question"):
            raise ValueError(f"{key}: 两边题目或 grounding_tier 不一致")
        for label, rows, judged in zip(("A", "B"), (a, b), scores):
            row, score = rows[key], judged.get(key, {})
            usage = row.get("usage") or {}
            pair[label] = {
                "l1_correct": row["l1_correct"], "execution_error": row.get("execution_error"),
                "answer_finish_reason": row.get("answer_finish_reason"),
                "seconds": row.get("seconds"), "cost_usd": row.get("cost_usd"),
                "answer_chars": len(row["final_answer"]) if "final_answer" in row else None,
                **{f"usage_{field}": usage.get(field) for field in usage_fields},
                "judge_error": {"stage": score.get("error_stage"), "message": score["error"]} if "error" in score else None,
                "score": None if "error" in score else score.get("score"),
                "final_judgement": None if "error" in score else score.get("final_judgement"),
                "retrieval": _retrieval_metrics(row),
            }
        pairs.append(pair)

    def summarize(selected: list) -> dict:
        metrics = {}
        for field in ("seconds", "cost_usd", "answer_chars", "score", *(f"usage_{f}" for f in usage_fields), "top1_distance", "evidence_chars"):
            #1.同一题先对有效配对求均值，bootstrap只抽独立题目
            by_question = {}
            for pair in selected:
                if field in ("top1_distance", "evidence_chars"):
                    if pair["grounding_tier"] != "must_ground":
                        continue
                    x, y = pair["A"]["retrieval"][field], pair["B"]["retrieval"][field]
                else:
                    x, y = pair["A"][field], pair["B"][field]
                if x is not None and y is not None:
                    by_question.setdefault(pair["sample_id"], []).append((x, y))
            n_observations = sum(len(rows) for rows in by_question.values())
            values = [(sum(x for x,y in rows) / len(rows), sum(y for x,y in rows) / len(rows))
                      for rows in by_question.values()]
            deltas = [y - x for x, y in values]
            metrics[field] = {
                "overall_n": sum(p["grounding_tier"] == "must_ground" for p in selected) if field in ("top1_distance", "evidence_chars") else len(selected),
                "n": n_observations, "n_questions": len(values),
                "A_mean": sum(x for x, y in values) / len(values) if values else None,
                "B_mean": sum(y for x, y in values) / len(values) if values else None,
                "mean_delta_B_minus_A": sum(deltas) / len(deltas) if deltas else None,
                "delta_ci95": bootstrap_ci(deltas) if deltas else None,
            }
        binary = {}
        for field in ("l1_correct", "final_judgement"):
            #2.McNemar按轮计算，多轮总数只描述观测，不合并成一次显著性检验
            rounds, question_ids = {}, set()
            for pair in selected:
                values = rounds.setdefault(pair["repeat"], [])
                x, y = pair["A"][field], pair["B"][field]
                if x is not None and y is not None:
                    values.append((x == "correct", y == "correct") if field == "final_judgement" else (x, y))
                    question_ids.add(pair["sample_id"])
            by_repeat = []
            for repeat, values in sorted(rounds.items()):
                only_a = sum(x and not y for x, y in values)
                only_b = sum(y and not x for x, y in values)
                by_repeat.append({"repeat": repeat, "n": len(values), "A_correct": sum(x for x,y in values),
                                  "B_correct": sum(y for x,y in values), "A_only": only_a, "B_only": only_b,
                                  "mcnemar_p": mcnemar_p(only_a, only_b) if values else None})
            binary[field] = {"overall_n": len(selected), "n_questions": len(question_ids),
                             **{key: sum(row[key] for row in by_repeat) for key in ("n", "A_correct", "B_correct", "A_only", "B_only")},
                             "mcnemar_p": by_repeat[0]["mcnemar_p"] if len(by_repeat) == 1 else None,
                             "by_repeat": by_repeat}
        return {"n": len(selected), "n_questions": len({p["sample_id"] for p in selected}),
                "must_ground_n": sum(p["grounding_tier"] == "must_ground" for p in selected),
                "execution_errors": {s: sum(bool(p[s]["execution_error"]) for p in selected) for s in ("A", "B")},
                "judge_errors": {s: sum(p[s]["judge_error"] is not None for p in selected) for s in ("A", "B")},
                "truncated_answers": {s: sum(p[s]["answer_finish_reason"] == "length" for p in selected) for s in ("A", "B")},
                "must_ground_retrieval": {
                    s: {status: sum(p["grounding_tier"] == "must_ground" and p[s]["retrieval"]["status"] == status for p in selected)
                        for status in ("searched", "not_searched", "missing_context", "invalid_context")}
                    for s in ("A", "B")
                },
                "binary": binary, "metrics": metrics}

    plan = json.loads(review_plan.read_text(encoding="utf-8")) if review_plan else None
    random_ids = plan["random_sample_ids"] if plan else []
    if plan and "population_ids" in plan and set(plan["population_ids"]) != {p["sample_id"] for p in pairs}:
        raise ValueError("回答题号与预先登记的 population_ids 不一致")
    if len(random_ids) != len(set(random_ids)) or set(random_ids) - {p["sample_id"] for p in pairs}:
        raise ValueError("人工抽样名单有重复或包含未评测题号")
    review_queue = []
    for pair in pairs:
        reasons = ["random"] if pair["sample_id"] in random_ids else []
        if all(pair[s]["final_judgement"] is not None for s in ("A", "B")) and pair["A"]["final_judgement"] != pair["B"]["final_judgement"]:
            reasons.append("judgement_disagreement")
        if any(pair[s]["answer_finish_reason"] == "length" for s in ("A", "B")):
            reasons.append("truncated")
        for error_field in ("judge_error", "execution_error"):
            if any(pair[s][error_field] is not None for s in ("A", "B")):
                reasons.append(error_field)
        if reasons:
            review_queue.append({"sample_id": pair["sample_id"], "repeat": pair["repeat"], "selection": reasons, "human_review": "pending"})
    result = {
        "inputs": {"A": str(path_a.resolve()), "B": str(path_b.resolve()), "judges": [str(p.resolve()) for p in judges or []]},
        "delta_direction": "B_minus_A", "usage_scope": "DeepSeek 全部调用合计；output 不是最终回答的单独 token 数",
        "inference_unit": "sample_id：连续指标先求每题均值再bootstrap；McNemar按repeat分轮计算",
        "subsets": {"all": summarize(pairs),
                    "known_pilot22": summarize([p for p in pairs if p["sample_id"] in PHASE1_JUDGE_SAMPLE_IDS]),
                    "remaining68": summarize([p for p in pairs if p["sample_id"] not in PHASE1_JUDGE_SAMPLE_IDS])},
        "pairs": pairs, "review_plan": plan, "review_queue": review_queue,
        "human_review_status": "pending" if review_queue else "not_requested",
        "publishable": bool(judges) and not any(p[s][field] is not None for p in pairs for s in ("A", "B") for field in ("judge_error", "execution_error")),
    }
    if out:
        review_path = out.with_name(f"{out.stem}.review.jsonl")
        map_path = out.with_name(f"{out.stem}.review-map.json")
        for path in ([out, review_path, map_path] if plan else [out]):
            if path.exists():
                raise FileExistsError(f"拒绝覆盖已有实验记录：{path}")
        out.parent.mkdir(parents=True, exist_ok=True)
        if plan:
            review_items = [(item, side) for item in review_queue for side in ("A", "B")]
            random.Random(plan["seed"]).shuffle(review_items)
            mappings = []
            with review_path.open("x", encoding="utf-8") as fp:
                for i, (item, side) in enumerate(review_items, 1):
                    row = (a if side == "A" else b)[(item["sample_id"], item["repeat"])]
                    review_id = f"r{i:03d}"
                    card = {"review_id": review_id, "question": row["question"], "final_answer": row["final_answer"],
                            "actual_tool_actions": row.get("tool_actions"), "tool_contexts": row.get("tool_contexts"),
                            "answer_finish_reason": row.get("answer_finish_reason"), "human_review": "pending"}
                    fp.write(json.dumps(card, ensure_ascii=False) + "\n")
                    mappings.append({"review_id": review_id, **item, "chain": side})
            with map_path.open("x", encoding="utf-8") as fp:
                json.dump(mappings, fp, ensure_ascii=False, indent=2)
                fp.write("\n")
            result["review_files"] = {"blinded": str(review_path.resolve()), "mapping": str(map_path.resolve())}
        with out.open("x", encoding="utf-8") as fp:
            json.dump(result, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
    metrics = result["subsets"]["all"]["metrics"]
    for side, path in (("A", path_a), ("B", path_b)):
        cost, seconds = metrics["cost_usd"][f"{side}_mean"], metrics["seconds"][f"{side}_mean"]
        cost_text = f"${cost:.5f}" if cost is not None else "缺失"
        seconds_text = f"{seconds:.1f} 秒" if seconds is not None else "缺失"
        print(f"{side} {path.name}: 每题 {cost_text} | {seconds_text}")
    for name, summary in result["subsets"].items():
        route = summary["binary"]["l1_correct"]
        p_text = "按轮报告" if len(route["by_repeat"]) > 1 else route["mcnemar_p"]
        print(f"{name}: {summary['n']} 对（{summary['n_questions']}题） | 路由正确 A={route['A_correct']} B={route['B_correct']} | McNemar p={p_text}")
        if len(route["by_repeat"]) > 1:
            for row in route["by_repeat"]:
                print(f"  repeat={row['repeat']}: {row['n']} 对 | McNemar p={row['mcnemar_p']}")
    if judges:
        summary = result["subsets"]["all"]
        print(f"评分有效配对 {metrics['score']['n']}/{len(pairs)} | 评分失败 {summary['judge_errors']} | 回答截断 {summary['truncated_answers']}")
        if not result["publishable"]:
            print("存在运行或评分失败，本轮不能作为完整成绩发布；失败记录已保留。")
    if out:
        print(f"配对统计和待人工复核名单已写入 {out}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="RepoRouter E2E pipeline eval (JSONL trace)")
    parser.add_argument("--limit", type=int, default=0, help="0 = 跑 manifest 全部样本")
    parser.add_argument("--phase1", action="store_true", help="按既定顺序跑 22 题试评，不能与 --limit / --bucket 联用")
    parser.add_argument(
        "--bucket",
        type=str,
        default="all",
        choices=["all", "domain", "far_general", "term_near_no_rag", "misleading_edge"],
    )
    #默认输出改到 data/v2/，不会覆盖 3 月的 data/e2e_eval_log*.jsonl
    parser.add_argument("--jsonl-out", type=Path, default=None, help="输出 JSONL，拒绝覆盖已有文件")
    parser.add_argument("--soft-fuse", action="store_true", help="启用 config 软熔断（需阈值）")
    parser.add_argument("--soft-fuse-threshold", type=float, default=1.12)
    parser.add_argument("--model", default=settings.model_name, help="deepseek-flash / deepseek-v4-pro")
    parser.add_argument("--max-tokens", type=int, default=800, help="每次回答调用的输出 token 上限")
    parser.add_argument("--thinking", action="store_true", help="打开 DeepSeek 思考模式")
    parser.add_argument("--routing-backend", default=settings.routing_backend, choices=["native", "phase2a"])
    parser.add_argument("--repeat", type=int, default=1, help="同一批题重复跑几次（大模型有随机性）")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("A.jsonl", "B.jsonl"), help="不跑模型，只对比两份结果（McNemar）")
    parser.add_argument("--judges", nargs=2, type=Path, help="与 --compare 对应的两份评分文件")
    parser.add_argument("--compare-out", type=Path, help="保存配对统计和人工复核队列，拒绝覆盖旧文件")
    parser.add_argument("--review-plan", type=Path, help="开跑前固定的人工抽样 JSON，含 random_sample_ids")
    args = parser.parse_args()
    if args.compare:
        _compare(*args.compare, judges=args.judges, out=args.compare_out, review_plan=args.review_plan)
        return
    if args.judges or args.compare_out or args.review_plan:
        parser.error("--judges / --compare-out / --review-plan 需与 --compare 联用")
    if args.repeat < 1 or args.max_tokens < 1 or args.limit < 0:
        parser.error("--repeat / --max-tokens 必须大于 0，--limit 不能为负数")
    if args.phase1 and (args.limit or args.bucket != "all"):
        parser.error("--phase1 不能与 --limit / --bucket 联用")
    if args.jsonl_out is None:
        tag = "phase2a" if args.routing_backend == "phase2a" else args.model + ("-thinking" if args.thinking else "")
        if args.phase1:
            tag += "_phase1"
        args.jsonl_out = ROOT / "data" / "v2" / f"e2e_{tag}.jsonl"
    if args.jsonl_out.exists():
        parser.error(f"输出文件已存在，请用 --jsonl-out 指定新文件：{args.jsonl_out}")

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
    per_rep_must = [0] * args.repeat
    per_rep_no = [0] * args.repeat
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
        with jsonl_path.open("x", encoding="utf-8") as fp_out:
            #同一批题按顺序重复跑 repeat 次
            for completed, (rep, row) in enumerate(((r, m) for r in range(args.repeat) for m in manifest), 1):
                q = row["query"]
                trace: list = []
                final_text = ""
                execution_error = None
                started = perf_counter()
                try:
                    agent = RepoRouter()
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
                per_rep_no[rep] += expect == "no" and ok_l1
                #每次工具调用都按 schema 查一遍参数（借鉴 BFCL 的 AST 检查）
                for ev in trace:
                    if "tool_name" in ev and "arguments" in ev:
                        args_total += 1
                        args_ok += check_tool_args(ev["tool_name"], ev.get("arguments"))
                per_rep_correct[rep] += ok_l1

                #本题的花费、耗时、缓存命中（core.chat 写在 trace 最后的 usage 事件里）
                usage = next((ev for ev in trace if ev.get("event") == "usage"), {})
                tool_actions = [{key: ev[key] for key in ("tool_name", "arguments", "error") if key in ev}
                                for ev in trace if "tool_name" in ev and "arguments" in ev]
                costs.append(usage.get("cost_usd", 0.0))
                secs.append(elapsed)
                cache_hit += usage.get("cache_hit", 0)
                cache_miss += usage.get("cache_miss", 0)

                gt = row.get("grounding_tier")
                if gt == "must_ground":
                    must_n += 1
                    if execution_error is None and actual_call:
                        must_ok += 1
                        per_rep_must[rep] += 1
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
                    "usage": usage or None,
                    "seconds": elapsed,
                    "router_decision": next((ev["decision"] for ev in trace if ev.get("event") == "router_decision"), None),
                    "bucket_type": row["bucket_type"],
                    "grounding_tier": row.get("grounding_tier"),
                    "should_call_tool": row["should_call_tool"],
                    "actual_call_search_codebase": actual_call,
                    "tool_names_round": _tool_names_from_trace(trace),
                    "search_arguments": [action["arguments"] for action in tool_actions if action["tool_name"] == "search_codebase"],
                    "tool_actions": tool_actions,
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
    print(f"\n配置：model={args.model} thinking={args.thinking} routing={args.routing_backend} repeat={args.repeat}")
    print("\n=== 第一层：Tool Decision（search_codebase）===")
    print(f"  样本数: {n}（{len(manifest)} 题 × {args.repeat} 次）")
    print(f"  运行失败: {execution_errors}（单独记录，不计入正确数或 FP/FN）")
    #重复观测只汇总次数，Wilson区间每轮单独算，不能把同一题算成多道题
    interval = "（重复观测汇总，95%CI按轮报告）"
    if args.repeat == 1:
        lo, hi = wilson_ci(l1_correct, n)
        interval = f"95%CI [{lo*100:.1f}, {hi*100:.1f}]"
    print(f"  决策与预期一致: {l1_correct}/{n}  ({(l1_correct/n*100 if n else 0):.1f}%)  {interval}")
    if args.repeat > 1:
        print(f"  每次分别: {per_rep_correct}（最少 {min(per_rep_correct)}，最多 {max(per_rep_correct)}）")
        for rep, correct in enumerate(per_rep_correct):
            lo, hi = wilson_ci(correct, len(manifest))
            print(f"  repeat={rep}: {correct}/{len(manifest)}  95%CI [{lo*100:.1f}, {hi*100:.1f}]")
    print(f"  False Positive（不该调却调）: {l1_fp}")
    print(f"  False Negative（该调未调）: {l1_fn}")
    print("  —— 子集 grounding_tier=must_ground（应检索）——")
    print(f"  must_ground 条数: {must_n}  |  已检索: {must_ok}  |  未检索(FN): {must_fn}")
    if must_n:
        print(f"  must_ground 检索率: {100.0 * must_ok / must_n:.1f}%")

    #借鉴 BFCL：把“该调时调了”和“不该调时没调”分开报，比一个混在一起的一致率更清楚
    print("  —— BFCL 口径 ——")
    for name, k, m, counts in (("相关检测（该查库时查了）", must_ok, must_n, per_rep_must), ("无关检测（不该查库时没查）", no_correct, no_n, per_rep_no)):
        if m:
            if args.repeat == 1:
                lo2, hi2 = wilson_ci(k, m)
                print(f"  {name}: {k}/{m} ({100.0 * k / m:.1f}%)  95%CI [{lo2*100:.1f}, {hi2*100:.1f}]")
            else:
                print(f"  {name}: {k}/{m} ({100.0 * k / m:.1f}%)（重复观测汇总）")
                for rep, correct in enumerate(counts):
                    lo2, hi2 = wilson_ci(correct, m // args.repeat)
                    print(f"    repeat={rep}: {correct}/{m // args.repeat}  95%CI [{lo2*100:.1f}, {hi2*100:.1f}]")
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
