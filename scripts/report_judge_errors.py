"""生成 data/reports/e2e_llm_judge_badcase_report*.md（默认 round2）。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.e2e_manifest import build_e2e_manifest
from src.evaluation.llm_judge_e2e import load_jsonl_records, is_bad_answer


def _fence_text(s: str) -> str:
    """用足够长的 Markdown fence 包住正文，避免内部 ``` 冲突。"""
    fence = "```"
    while fence in s:
        fence += "`"
    return f"{fence}text\n{s}\n{fence}"


def main() -> None:
    ap = argparse.ArgumentParser(description="导出 LLM Judge bad case Markdown（含 E2E 模型回答）")
    ap.add_argument(
        "--judge-jsonl",
        type=Path,
        default=ROOT / "data" / "e2e_llm_judge_full_round2.jsonl",
        help="LLM Judge 输出 JSONL（第二轮默认 e2e_llm_judge_full_round2.jsonl）",
    )
    ap.add_argument(
        "--eval-jsonl",
        type=Path,
        default=ROOT / "data" / "e2e_eval_log_promptv2.jsonl",
        help="与 Judge 同批次的 E2E 日志（优先完整回答，兼容历史摘要）",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "reports" / "e2e_llm_judge_badcase_report_round2.md",
        help="输出 Markdown 路径",
    )
    args = ap.parse_args()
    if args.out.exists() or args.out.is_symlink():
        ap.error(f"输出文件已存在，请用 --out 指定新文件：{args.out}")

    man = {r["id"]: r for r in build_e2e_manifest()}
    gt = {k: v.get("grounding_tier") for k, v in man.items()}
    gt_ia = {k: v.get("grounding_tier_index_aware") for k, v in man.items()}
    vpath = ROOT / "data" / "chroma_verified_skip_must_ground.json"
    verified_payload: dict = {}
    verified: dict = {}
    if vpath.is_file():
        verified_payload = json.loads(vpath.read_text(encoding="utf-8"))
        verified = verified_payload.get("entries", {})
    p = args.judge_jsonl.resolve()
    if not p.is_file():
        print(f"找不到 Judge 文件: {p}", file=sys.stderr)
        sys.exit(1)
    rows = list(load_jsonl_records(p).values())
    eval_by_id = load_jsonl_records(args.eval_jsonl.resolve()) if args.eval_jsonl.is_file() else {}
    #1.有来源指纹时先核对，避免把另一批回答配给同名题号
    if args.eval_jsonl.is_file():
        digest = hashlib.sha256(args.eval_jsonl.read_bytes()).hexdigest()
        if any(r.get("input_sha256") and r["input_sha256"] != digest for r in rows):
            ap.error("Judge 的 input_sha256 与所给 E2E 日志不一致")
    rows = [r for r in rows if r.get("final_judgement") and not r.get("error")]
    bad = [r for r in rows if is_bad_answer(r)]

    eval_note = ""
    if eval_by_id:
        ep = args.eval_jsonl.resolve()
        try:
            ep_disp = ep.relative_to(ROOT).as_posix()
        except ValueError:
            ep_disp = ep.as_posix()
        eval_note = (
            f"**模型回答**来自同期 E2E 日志 `{ep_disp}`，按 `(sample_id, repeat)` 配对。"
            "优先使用 `final_answer`；旧日志使用可能截断的 `final_answer_preview`。"
        )
    else:
        eval_note = (
            f"⚠ 未找到 E2E 日志 `{args.eval_jsonl}`，**未附模型回答**；"
            "请将 `--eval-jsonl` 指到与 Judge 同批次的 `e2e_eval_log*.jsonl` 后重跑。"
        )

    n_inc = len([x for x in bad if x["final_judgement"] == "incorrect"])
    n_part = len([x for x in bad if x["final_judgement"] == "partially_correct"])
    jp = p
    try:
        jp_disp = jp.relative_to(ROOT).as_posix()
    except ValueError:
        jp_disp = jp.as_posix()

    chroma_note = ""
    if verified_payload:
        sqlite_disp = verified_payload.get("chroma_sqlite", "历史记录未提供数据库路径")
        chroma_note = (
            "**Chroma 验证**：`data/chroma_verified_skip_must_ground.json`（"
            "由 `scripts/update_grounding_labels.py` 读取 **`embedding_metadata`** 全文并与 `src/` 对照）；"
            f" 数据库文件：**`{sqlite_disp}`**（embeddings={verified_payload.get('embeddings_count', '?')}）。"
            "「index-aware tier」列为 manifest 合并后的 gold。"
        )

    lines: list[str] = [
        f"# LLM Judge bad case 汇总（{len(rows)} 条有效评分）",
        "",
        f"**Judge 来源**：`{jp_disp}`",
        "",
        "**定义**：`final_judgement` ≠ `correct`，或 `grounding_appropriate`=false，或 `context_pollution`=true。",
        f"**共 {len(bad)} 条 bad case**；`repeat` 从 0 开始，旧日志缺省为 0。",
        "",
        eval_note,
        "",
        chroma_note,
        "",
        f"## 一、`incorrect`（{n_inc} 条）",
        "",
        "| id | repeat | grounding_tier | index-aware tier | Chroma 验证 | score | 检索 | 软熔断 | top1 | 污染 |",
        "|---|---:|---|----|----|--:|:---|:---|:---|:---|",
    ]
    for r in sorted([x for x in bad if x["final_judgement"] == "incorrect"], key=lambda x: (x["sample_id"], x.get("repeat", 0))):
        sid = r["sample_id"]
        vrec = verified.get(sid, {})
        vshort = ""
        if vrec:
            if vrec.get("reason") == "scripts_scope":
                vshort = "scripts 不在库"
            elif vrec.get("missing_tokens_in_chroma"):
                vshort = "src 有、库无: " + ", ".join(vrec["missing_tokens_in_chroma"][:4])
            else:
                vshort = vrec.get("reason", "")
        lines.append(
            f"| {sid} | {r.get('repeat', 0)} | {gt.get(sid, '')} | {gt_ia.get(sid, '')} | {vshort} | {r['score']} | "
            f"{r.get('actual_call_search_codebase')} | {r.get('soft_fallback_triggered')} | "
            f"{r.get('top1_distance')} | {r.get('context_pollution')} |"
        )
    lines += ["", f"## 二、`partially_correct`（{n_part} 条）", ""]
    lines += [
        "| id | repeat | grounding_tier | index-aware tier | Chroma 验证 | score | 检索 | 软熔断 | top1 | 污染 |",
        "|---|---:|---|----|----|--:|:---|:---|:---|:---|",
    ]
    for r in sorted([x for x in bad if x["final_judgement"] == "partially_correct"], key=lambda x: (x["sample_id"], x.get("repeat", 0))):
        sid = r["sample_id"]
        vrec = verified.get(sid, {})
        vshort = ""
        if vrec:
            if vrec.get("reason") == "scripts_scope":
                vshort = "scripts 不在库"
            elif vrec.get("missing_tokens_in_chroma"):
                vshort = "src 有、库无: " + ", ".join(vrec["missing_tokens_in_chroma"][:4])
            else:
                vshort = vrec.get("reason", "")
        lines.append(
            f"| {sid} | {r.get('repeat', 0)} | {gt.get(sid, '')} | {gt_ia.get(sid, '')} | {vshort} | {r['score']} | "
            f"{r.get('actual_call_search_codebase')} | {r.get('soft_fallback_triggered')} | "
            f"{r.get('top1_distance')} | {r.get('context_pollution')} |"
        )
    lines += ["", "## 三、逐条（题干 + 模型回答 + Judge reason）", ""]
    #2.同题不同轮分别展示，缺少对应回答时不能借用另一轮
    for r in sorted(bad, key=lambda x: (x["final_judgement"], x["sample_id"], x.get("repeat", 0))):
        sid = r["sample_id"]
        repeat = r.get("repeat", 0)
        ev = eval_by_id.get((sid, repeat), {})
        lines.append(f"### {sid} · repeat={repeat}（{r['final_judgement']}，score={r['score']}）")
        lines.append("")
        q = r.get("question") or ev.get("question") or ""
        lines.append("**题干**")
        lines.append("")
        lines.append(q)
        lines.append("")
        ans = ev.get("final_answer", ev.get("final_answer_preview", "")) or ""
        if ans:
            lines.append("**模型回答**")
            lines.append("")
            lines.append(_fence_text(ans))
        else:
            lines.append("**模型回答**：（本日志中没有对应轮次的回答正文）")
        lines.append("")
        lines.append("**Judge reason**")
        lines.append("")
        lines.append(r.get("reason", ""))
        lines.append("")
        vr = verified.get(sid)
        if vr:
            lines.append("**Chroma 验证（sqlite 全文 × src）**")
            lines.append("")
            lines.append(f"- suggested tier: **{vr.get('tier')}**（{vr.get('reason')}）")
            if vr.get("detail"):
                lines.append(f"- {vr['detail']}")
            if vr.get("missing_tokens_in_chroma"):
                lines.append(f"- 库中未出现符号: `{vr['missing_tokens_in_chroma']}`")
        lines.append("")

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
