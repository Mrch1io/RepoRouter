"""
对 E2E JSONL 跑 LLM-as-a-Judge（第三层）。

评委换成百炼上的 Qwen（judge_model，默认 qwen3.8-max），和被评的 DeepSeek 不是同一家，避免“自己给自己打分”。
默认输出到 data/v2/judge_<输入文件名>.jsonl，不覆盖 3 月的结果。

用法（项目根，需 .env 中 qwen_api_key）：
  python scripts/judge_answers.py --jsonl data/v2/e2e_deepseek-flash.jsonl --full --workers 4
  python scripts/judge_answers.py --jsonl data/e2e_eval_log_promptv2.jsonl
  python scripts/judge_answers.py --jsonl data/e2e_eval_log_promptv2.jsonl --full --out data/eval/e2e_llm_judge_full.jsonl
  # 使用 manifest 的 grounding_tier_index_aware 作为 Judge Gold（与 chroma 可达性对齐），**新文件**不覆盖 round2
  python scripts/judge_answers.py --jsonl data/e2e_eval_log_promptv2.jsonl --full --index-aware
  python scripts/judge_answers.py --jsonl data/e2e_eval_log_promptv2.jsonl --dry-run
  # 并发调用 Judge API（默认 1；大评测可试 4～8，视 DeepSeek 限流调整）
  python scripts/judge_answers.py --jsonl data/e2e_eval_log_promptv2.jsonl --full --index-aware --workers 6
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from src.config import settings
from src.evaluation.e2e_manifest import build_e2e_manifest
from src.evaluation.llm_judge_e2e import (
    PHASE1_JUDGE_SAMPLE_IDS,
    all_manifest_sample_ids,
    build_judge_messages,
    parse_judge_json,
    validate_judge_output,
)


#评委走百炼的 OpenAI 兼容接口
_judge_client = OpenAI(api_key=settings.qwen_api_key or "missing-key", base_url=settings.qwen_base_url, timeout=120)


def chat_completion(messages, max_tokens: int, temperature: float, model: str):
    #评委是 deepseek-* 时走 DeepSeek 接口（当第二评委：看两家评委是否一致、DeepSeek 给自家打分会不会偏高）
    #复用 src/agent/llm.py：非思考模式，顺带累计花费
    if model.startswith("deepseek"):
        from src.agent.llm import chat_completion as deepseek_chat
        return deepseek_chat(messages, max_tokens=max_tokens, temperature=temperature, model=model)
    resp = _judge_client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
    )
    return resp.choices[0].message


def _load_jsonl_by_id(path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            #重复跑多次的结果只评第 1 次，和 3 月的 90 条口径一致
            if rec.get("repeat", 0) == 0:
                if rec["sample_id"] in by_id:
                    raise ValueError(f"{path}: 重复的 repeat=0 记录 {rec['sample_id']}")
                by_id[rec["sample_id"]] = rec
    return by_id


def main() -> None:
    p = argparse.ArgumentParser(description="LLM-as-judge on E2E JSONL (phase1 sample or full 90)")
    p.add_argument("--jsonl", type=Path, required=True, help="e2e_eval_log*.jsonl")
    p.add_argument(
        "--full",
        action="store_true",
        help="评测 manifest 全量 90 条（覆盖默认的 Phase1 抽样 22 条）",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 JSONL（默认：phase1 -> data/eval/e2e_llm_judge_phase1.jsonl；--full -> data/eval/e2e_llm_judge_full.jsonl）",
    )
    p.add_argument("--dry-run", action="store_true", help="只打印 prompt 条数，不调 API")
    p.add_argument(
        "--index-aware",
        action="store_true",
        help="Judge prompt 使用 grounding_tier_index_aware（需已生成 data/chroma_verified_skip_must_ground.json 等合并进 manifest）",
    )
    p.add_argument("--judge-model", default=settings.judge_model)
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="并发线程数（仅非 dry-run）。每条 Judge 相互独立；>1 可显著缩短总耗时，但需遵守 API 限流（建议 4～8 试起）",
    )
    args = p.parse_args()
    if args.workers < 1:
        p.error("--workers 必须大于 0")

    sample_ids = all_manifest_sample_ids() if args.full else PHASE1_JUDGE_SAMPLE_IDS
    manifest_by_id = {r["id"]: r for r in build_e2e_manifest()}
    out_path = args.out
    if out_path is None:
        out_path = ROOT / "data" / "v2" / f"judge_{args.jsonl.stem}{'_index_aware' if args.index_aware else ''}.jsonl"
        #换了评委就在文件名后面带上评委名，不覆盖默认评委（qwen3.8-max）的结果
        if args.judge_model != settings.judge_model:
            out_path = out_path.with_name(f"{out_path.stem}__by_{args.judge_model}.jsonl")

    if not args.jsonl.is_file():
        print(f"文件不存在: {args.jsonl}", file=sys.stderr)
        sys.exit(1)

    input_path = args.jsonl.resolve()
    if out_path.resolve() == input_path:
        p.error("输出文件不能覆盖输入回答日志")
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    by_id = _load_jsonl_by_id(input_path)
    missing = [s for s in sample_ids if s not in by_id]
    if missing:
        print(f"JSONL 中缺少 sample_id（共 {len(missing)} 个）: {missing[:20]}...", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(
            f"将评测 {len(sample_ids)} 条（{'full 90' if args.full else 'phase1 22'}）"
            f"，index_aware={args.index_aware}，workers={args.workers}，out={out_path}"
        )
        for sid in sample_ids[:1]:
            msgs = build_judge_messages(by_id[sid], use_index_aware_tier=args.index_aware)
            print("--- user Gold 片段（含 tier）---\n", msgs[1]["content"][:900], "...\n")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _judge_one(sid: str) -> dict:
        rec = by_id[sid]
        metadata = {
            "sample_id": sid,
            "judge_model": args.judge_model,
            "judge_max_tokens": args.max_tokens,
            "judge_temperature": args.temperature,
            "judge_prompt_version": "full_answer_v1",
            "input_jsonl": str(input_path),
            "input_sha256": input_sha256,
            "answer_format": "full" if "final_answer" in rec else "legacy_excerpt",
            "answer_finish_reason": rec.get("answer_finish_reason"),
            "judge_gold_mode": "index_aware" if args.index_aware else "legacy",
        }
        if rec.get("execution_error"):
            return {**metadata, "error_stage": "answer_execution", "error": rec["execution_error"]}
        msg = None
        stage = "judge_input"
        try:
            messages = build_judge_messages(rec, use_index_aware_tier=args.index_aware)
            stage = "judge_request"
            msg = chat_completion(
                messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                model=args.judge_model,
            )
            stage = "judge_output"
            raw = (msg.content or "").strip()
            obj = parse_judge_json(raw)
            validate_judge_output(obj)
        except Exception as e:
            return {
                **metadata,
                "error_stage": stage,
                "error": str(e),
                "raw_judge_output": (msg.content or "") if msg else "",
            }
        m = manifest_by_id.get(sid, {})
        gt_legacy = m.get("grounding_tier")
        gt_ia = m.get("grounding_tier_index_aware", gt_legacy)
        if args.index_aware:
            gt_out = gt_ia
        else:
            gt_out = rec.get("grounding_tier") or gt_legacy
        out: dict = {
            **metadata,
            "bucket_type": rec.get("bucket_type"),
            "grounding_tier": gt_out,
            "question": rec.get("question"),
            "actual_call_search_codebase": rec.get("actual_call_search_codebase"),
            "top1_distance": rec.get("top1_distance"),
            "soft_fallback_triggered": rec.get("soft_fallback_triggered"),
            **{key: obj[key] for key in (
                "answer_relevant", "grounding_appropriate", "context_pollution",
                "score", "final_judgement", "reason",
            )},
        }
        if args.index_aware:
            out["grounding_tier_original"] = gt_legacy
        return out

    ok = 0
    err = 0
    with out_path.open("w", encoding="utf-8") as fp, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_judge_one, sid) for sid in sample_ids]
        for future in as_completed(futures):
            row = future.result()
            sid = row["sample_id"]
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            fp.flush()
            if "error" in row:
                err += 1
                print(f"[fail] {sid} ({row['error_stage']}): {row['error']}", file=sys.stderr, flush=True)
                continue
            ok += 1
            fj = row.get("final_judgement")
            sc = row.get("score")
            print(f"[ok] {ok}/{len(sample_ids)} {sid} -> {fj} score={sc}", flush=True)

    print(f"\n完成: 成功 {ok}, 失败 {err}, 写入 {out_path}")
    if args.judge_model.startswith("deepseek"):
        from src.agent.llm import USAGE
        print(f"评委花费约 ${USAGE['cost_usd']:.4f}（{USAGE['calls']} 次调用）")


if __name__ == "__main__":
    main()
