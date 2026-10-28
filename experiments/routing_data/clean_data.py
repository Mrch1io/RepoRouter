"""
Phase 2A 训练 JSONL 清洗审计：对照 CLEANING_CHECKLIST + LABELING_RULES。

- 写出带自动修补的副本（默认仅同步 bucket ↔ _meta.slice / slice_label_zh）。
- 写出 Markdown 报告：ERROR / WARN / REVIEW（需人工）分级。
- 不覆盖已有输出；发现 ERROR 时只写报告，不输出清洗后的数据。

用法（仓库根目录）：
  python experiments/routing_data/clean_data.py
  python experiments/routing_data/clean_data.py --input experiments/routing_data/outputs/ft_qa_500_train.jsonl
  python experiments/routing_data/clean_data.py --extract-review
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
import copy
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.agent.tools.schemas import TOOLS_SCHEMA, check_tool_args

ALLOWED_TOOLS = frozenset(tool["function"]["name"] for tool in TOOLS_SCHEMA)

BUCKET_TO_SLICE: dict[str, tuple[str, str]] = {
    "domain": ("private_domain", "私域语境（内部项目 ModelHub-Serve，基于 FastAPI+Transformers）"),  # 和生成脚本同步
    "far_general": ("dev_general_no_rag", "开发通识（无需查库）"),
    "misleading_edge": ("misleading_edge", "误导向/边界"),
}

# 可疑：无明确任务请求（清洗清单 F + 非任务型）
NON_TASK_QUERY_PATTERNS = [
    re.compile(r"^ping\.?$", re.I),
    re.compile(r"^who are you\??$", re.I),
    re.compile(r"^hi[,!]?\s*$", re.I),
    re.compile(r"^hello\??$", re.I),
]

# 泛通识 / 选型对比（B1：domain 桶可能应迁 far_general）
GENERIC_KNOWLEDGE_HINTS = re.compile(
    r"(什么是|有哪些|区别|相比|优势|劣势|更快吗|还是|哪个好|"
    r"Git\s|rebase|merge|分支|概念|入门|介绍|官方文档|"
    r"https?://|\.com/|URL|首页)",
    re.I,
)

# 伪 grounded：路径/镜像/版本号（clarify 文本红牌）
GROUNDING_RISK = re.compile(
    r"(/src/|/home/|\.py:\d+|:\d+:\d+|python:\d+\.\d+-alpine|"
    r"cuda\s*\d|docker\s+run|top_k\s*=\s*\d|咱们项目|主目录默认)",
    re.I,
)

# 表演性客套（F）
FLUFF = re.compile(r"(不好意思|非常愿意|亲～|呢~|么么|跪求)", re.I)


@dataclass
class Issue:
    severity: str  # ERROR | WARN | REVIEW
    code: str
    message: str


@dataclass
class RowAudit:
    sample_id: str
    issues: list[Issue] = field(default_factory=list)
    auto_fixed: list[str] = field(default_factory=list)


def _validate_schema(row: dict, *, issues: list[Issue]) -> None:
    #1.先检查结构，后面的规则才可以按字符串和字典读取。
    if not isinstance(row, dict):
        issues.append(Issue("ERROR", "bad_record", "每条记录须为对象"))
        return
    required = ("sample_id", "bucket", "query", "decision_type")
    for k in required:
        if k not in row or row[k] is None:
            issues.append(Issue("ERROR", "missing_key", f"缺少或为空: {k}"))
        elif not isinstance(row[k], str) or not row[k].strip():
            issues.append(Issue("ERROR", "bad_field", f"{k} 须为非空字符串"))
    if issues:
        return
    if row.get("_meta") is not None and not isinstance(row["_meta"], dict):
        issues.append(Issue("ERROR", "bad_meta", "_meta 须为对象或 null"))
    if row.get("clarify_answer") is not None and not isinstance(row["clarify_answer"], str):
        issues.append(Issue("ERROR", "bad_clarify", "clarify_answer 须为字符串或 null"))

    if row.get("bucket") not in BUCKET_TO_SLICE:
        issues.append(Issue("ERROR", "bad_bucket", f"bucket 非法: {row.get('bucket')!r}"))

    dt = row.get("decision_type")
    if dt not in ("tool_call", "no_tool", "clarify_needed"):
        issues.append(Issue("ERROR", "bad_decision_type", f"decision_type 非法: {dt!r}"))

    if dt == "tool_call":
        name = row.get("tool_name")
        if not isinstance(name, str) or name not in ALLOWED_TOOLS:
            issues.append(Issue("ERROR", "bad_tool_name", f"tool_name 非法或缺失: {name!r}"))
        elif not check_tool_args(name,row.get("arguments")):
            #2.复用运行时工具定义，必填项、可选参数和范围只维护一份。
            issues.append(Issue("ERROR", "bad_arguments", f"工具 {name} 的参数不符合定义"))
        if row.get("clarify_answer") not in (None, ""):
            issues.append(Issue("WARN", "tool_call_clarify", "tool_call 时 clarify_answer 应为 null"))
    elif dt == "no_tool":
        for f in ("tool_name", "arguments", "clarify_answer", "oracle_context"):
            if row.get(f) is not None:
                issues.append(Issue("WARN", f"no_tool_{f}", f"no_tool 时 {f} 应为 null"))
    elif dt == "clarify_needed":
        ca = row.get("clarify_answer")
        if not ca or not str(ca).strip():
            issues.append(Issue("ERROR", "empty_clarify", "clarify_needed 须含非空 clarify_answer"))
        if row.get("tool_name") is not None or row.get("arguments") is not None:
            issues.append(Issue("WARN", "clarify_needs_null_tools", "clarify_needed 时 tool/arguments 应为 null"))


def _prefix_for_bucket(bucket: str) -> str:
    return {"domain": "domain_", "far_general": "far_general_", "misleading_edge": "misleading_edge_"}[bucket]


def _heuristic_review(row: dict, *, issues: list[Issue]) -> None:
    q = (row.get("query") or "").strip()
    sid = row.get("sample_id", "")
    bucket = row.get("bucket", "")
    dt = row.get("decision_type", "")
    meta = row.get("_meta") or {}

    if sid and bucket:
        exp = _prefix_for_bucket(bucket)
        if not sid.startswith(exp):
            issues.append(
                Issue(
                    "REVIEW",
                    "id_bucket_mismatch",
                    f"sample_id 前缀与 bucket 不一致: {sid!r} vs {bucket!r}",
                )
            )

    if len(q) < 6:
        issues.append(Issue("REVIEW", "very_short_query", f"query 极短（{len(q)} 字符），是否非任务型？"))

    for pat in NON_TASK_QUERY_PATTERNS:
        if pat.match(q.strip()):
            issues.append(Issue("REVIEW", "non_task_query", "可能非开发任务（问候/空 ping）"))

    if bucket == "domain" and dt == "no_tool" and GENERIC_KNOWLEDGE_HINTS.search(q):
        issues.append(
            Issue(
                "REVIEW",
                "b1_domain_generic",
                "domain 桶 + no_tool，但 query 像通识/选型；是否应迁 far_general（见 LABELING_RULES）",
            )
        )

    if bucket == "far_general" and re.search(
        r"(transformers/|fastapi/|src/transformers|huggingface|建模|APIRouter|pipeline\()", q, re.I
    ):
        issues.append(
            Issue(
                "REVIEW",
                "b2_far_repo_specific",
                "far_general 桶但 query 含强私域/库内术语，是否应 domain？",
            )
        )

    if dt == "clarify_needed":
        ca = row.get("clarify_answer") or ""
        if len(ca) > 220:
            issues.append(Issue("WARN", "long_clarify", "clarify_answer 偏长（>220），可能教程式；建议压到 1～2 句"))
        if FLUFF.search(ca):
            issues.append(Issue("WARN", "fluff_clarify", "clarify_answer 含表演性客套用语"))
        if GROUNDING_RISK.search(ca):
            issues.append(Issue("WARN", "grounding_in_clarify", "clarify_answer 含路径/镜像/版本等，核对是否伪 grounded"))

    if dt == "tool_call" and row.get("tool_name") == "analyse_log":
        args = row.get("arguments") or {}
        log = str(args.get("log") or "")
        if "traceback" not in log.lower() and "error" not in log.lower() and "exception" not in log.lower():
            if "traceback" not in q.lower() and "error" not in q.lower():
                issues.append(
                    Issue(
                        "REVIEW",
                        "analyse_log_weak",
                        "analyse_log 但 log 参数/题干中未必含典型报错关键词；核对是否应 search_codebase",
                    )
                )

    # 规则 5：通识题不查私有代码库（和运行时 system prompt、测试金标一致）
    if dt == "tool_call" and row.get("tool_name") == "search_codebase" and bucket == "far_general":
        issues.append(Issue("ERROR", "far_general_search", "通识切片不允许 search_codebase（见路由规则 5）；改 no_tool 或删"))

    if dt == "tool_call" and row.get("tool_name") == "search_codebase":
        args = row.get("arguments") or {}
        sq = str(args.get("query") or "").strip()
        if len(sq) < 4:
            issues.append(Issue("ERROR", "empty_search_query", "search_codebase 的 query 过短或为空"))
        elif len(sq) > 400:
            issues.append(Issue("WARN", "long_search_query", "search query 过长，可能应精炼"))

    if dt == "clarify_needed" and re.search(
        r"(在哪|在哪|找一下|搜索|定位|文件|实现|源码|类|函数)", q
    ) and not re.search(r"(哪个|是否|还是|还是)", q):
        issues.append(
            Issue(
                "REVIEW",
                "clarify_vs_search",
                "题干像明确检索意图，但标为 clarify_needed；若方向明确可改 search_codebase（见路由规则 3）",
            )
        )

    if meta.get("slice") and bucket:
        exp_slice, _ = BUCKET_TO_SLICE[bucket]
        if meta["slice"] != exp_slice:
            issues.append(
                Issue(
                    "WARN",
                    "meta_slice_bucket",
                    f"_meta.slice={meta['slice']!r} 与 bucket={bucket!r} 期望 {exp_slice!r} 不一致",
                )
            )


def collect_issues(row: dict) -> list[Issue]:
    """单条审计（不做 auto_fix），用于导出人工复核子集。"""
    issues: list[Issue] = []
    _validate_schema(row, issues=issues)
    if not any(issue.severity == "ERROR" for issue in issues):
        _heuristic_review(row, issues=issues)
    return issues


def _auto_fix_meta(row: dict) -> list[str]:
    fixed: list[str] = []
    bucket = row.get("bucket")
    if bucket not in BUCKET_TO_SLICE:
        return fixed
    exp_slice, exp_label = BUCKET_TO_SLICE[bucket]
    meta = dict(row.get("_meta") or {})
    if meta.get("slice") != exp_slice or meta.get("slice_label_zh") != exp_label:
        meta["slice"] = exp_slice
        meta["slice_label_zh"] = exp_label
        row["_meta"] = meta
        fixed.append("synced _meta.slice + slice_label_zh from bucket")
    return fixed


def _check_outputs(source : Path,outputs : list[Path])->None:
    #1.写任何文件前检查全部目标，连断开的符号链接也不能覆盖。
    paths = {source.resolve()}
    for output in outputs:
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"拒绝覆盖已有输出：{output}")
        target = output.resolve()
        if any(target == previous or target in previous.parents or previous in target.parents for previous in paths):
            raise ValueError(f"输入或输出路径冲突：{output}")
        paths.add(target)


def run(
    path: Path,
    *,
    out_jsonl: Path | None,
    out_report: Path,
    dry_run: bool,
) -> None:
    #1.先保护旧文件，再读取并审计；有错也继续检查后面的记录。
    _check_outputs(path, [out_report] + ([out_jsonl] if out_jsonl and not dry_run else []))
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    audits: list[RowAudit] = []
    stats: dict[str, int] = defaultdict(int)
    by_code: dict[str, list[str]] = defaultdict(list)

    cleaned_rows: list[dict] = []

    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
            issues = collect_issues(row)
        except json.JSONDecodeError as e:
            row = None
            issues = [Issue("ERROR", "bad_json", f"第 {number} 条 JSON 无法解析：{e.msg}")]
        sid = str(row.get("sample_id", "?")) if isinstance(row, dict) else f"第 {number} 条"
        row_copy = copy.deepcopy(row)
        auto_fixed: list[str] = []
        if not any(issue.severity == "ERROR" for issue in issues):
            if not dry_run:
                auto_fixed = _auto_fix_meta(row_copy)
                if auto_fixed:
                    issues = collect_issues(row_copy)
            cleaned_rows.append(row_copy)

        for iss in issues:
            stats[iss.severity] += 1
            by_code[iss.code].append(sid)
        audits.append(RowAudit(sample_id=sid, issues=issues, auto_fixed=auto_fixed))

    # Report markdown
    lines_out: list[str] = []
    lines_out.append("# Phase 2A 训练数据清洗报告\n")
    lines_out.append("\n## 执行摘要\n\n")
    lines_out.append(f"- **ERROR**（必须修）: {stats.get('ERROR', 0)} 条\n")
    lines_out.append(f"- **WARN**（建议修）: {stats.get('WARN', 0)} 条\n")
    lines_out.append(f"- **REVIEW**（人工复核）: {stats.get('REVIEW', 0)} 条\n")
    lines_out.append(
        "- 详见下方「需优先人工复核」表与「全量逐条」；附录含 decision_type / bucket / 工具分布与重复 query。\n"
    )
    lines_out.append("\n## 元信息\n\n")
    source = path.resolve()
    lines_out.append(f"- 源文件: `{source.relative_to(ROOT) if source.is_relative_to(ROOT) else source}`\n")
    lines_out.append(f"- 条数: {len(audits)}\n")
    if stats.get("ERROR", 0):
        lines_out.append("- 存在 ERROR，本次不输出清洗数据；分布统计仅包含通过校验的记录。\n")
    lines_out.append(
        "- 自动修补: `_meta.slice` / `slice_label_zh` 与 `bucket` 对齐（见 `*.cleaned.jsonl`）\n"
    )
    lines_out.append("\n## 汇总：按问题代码（sample_id 列表）\n")

    for code in sorted(by_code.keys()):
        ids = by_code[code]
        lines_out.append(f"\n### `{code}`（{len(ids)} 条）\n")
        lines_out.append(", ".join(f"`{x}`" for x in ids[:80]))
        if len(ids) > 80:
            lines_out.append(f"\n… 共 {len(ids)} 条\n")

    lines_out.append("\n## 需优先人工复核（REVIEW + ERROR）\n")
    lines_out.append("| sample_id | 级别 | 代码 | 说明 |\n")
    lines_out.append("|-----------|------|------|------|\n")
    for a in audits:
        for iss in a.issues:
            if iss.severity in ("ERROR", "REVIEW"):
                msg = iss.message.replace("|", "\\|")
                lines_out.append(f"| `{a.sample_id}` | {iss.severity} | `{iss.code}` | {msg} |\n")

    lines_out.append("\n## 全量逐条（含 WARN）\n")
    for a in audits:
        if not a.issues and not a.auto_fixed:
            continue
        lines_out.append(f"\n### `{a.sample_id}`\n")
        if a.auto_fixed:
            lines_out.append(f"- **auto**: {', '.join(a.auto_fixed)}\n")
        for iss in a.issues:
            lines_out.append(f"- **{iss.severity}** [`{iss.code}`]: {iss.message}\n")

    # 分布与重复 query
    dt_c = Counter(r.get("decision_type") for r in cleaned_rows)
    bk_c = Counter(r.get("bucket") for r in cleaned_rows)
    tool_c = Counter()
    for r in cleaned_rows:
        if r.get("decision_type") == "tool_call" and r.get("tool_name"):
            tool_c[r["tool_name"]] += 1
    qmap: dict[str, list[str]] = defaultdict(list)
    for r in cleaned_rows:
        q = (r.get("query") or "").strip()
        if q:
            qmap[q].append(r.get("sample_id", "?"))
    dup_queries = {q: ids for q, ids in qmap.items() if len(ids) > 1}

    stats_lines = [
        "\n---\n\n## 附录：分布统计\n\n",
        "| decision_type | 条数 |\n|---|---:|\n",
    ]
    for k in sorted(dt_c.keys()):
        stats_lines.append(f"| `{k}` | {dt_c[k]} |\n")
    stats_lines.append("\n| bucket | 条数 |\n|---|---:|\n")
    for k in sorted(bk_c.keys()):
        stats_lines.append(f"| `{k}` | {bk_c[k]} |\n")
    stats_lines.append("\n### tool_call 工具分布\n\n| tool_name | 条数 |\n|---|---:|\n")
    for k in sorted(tool_c.keys()):
        stats_lines.append(f"| `{k}` | {tool_c[k]} |\n")
    if dup_queries:
        stats_lines.append("\n### 完全重复 query（需人工判断是否保留多条）\n\n")
        for q, ids in sorted(dup_queries.items(), key=lambda x: -len(x[1]))[:30]:
            stats_lines.append(f"- {len(ids)} 条: " + ", ".join(f"`{i}`" for i in ids) + "\n")
            preview = q[:120].replace("\n", " ")
            stats_lines.append(f"  - 文本预览: {preview}{'…' if len(q)>120 else ''}\n")
    else:
        stats_lines.append("\n### 完全重复 query\n\n无。\n")

    #2.全部审完才写新文件，独占创建防止检查之后被其他进程抢先占用。
    with out_report.open("x", encoding="utf-8") as report:
        report.write("".join(lines_out) + "".join(stats_lines))
    print(f"Report -> {out_report}")
    if stats.get("ERROR", 0):
        print("发现 ERROR，未输出清洗后的数据，请先查看报告。")
    elif out_jsonl and not dry_run:
        with out_jsonl.open("x", encoding="utf-8") as output:
            output.write("\n".join(json.dumps(r, ensure_ascii=False) for r in cleaned_rows) + "\n")
        print(f"Cleaned -> {out_jsonl} ({len(cleaned_rows)} rows)")


def extract_manual_review(
    path: Path,
    out_path: Path,
) -> int:
    """写出含至少一条 REVIEW 的样本；每条增加 `_manual_review`: [{{code, message}}, ...]。"""
    #1.ERROR不能静默跳过，先修正结构再导出需要人工判断的样本。
    _check_outputs(path, [out_path])
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out_rows: list[dict] = []
    for line in lines:
        row = json.loads(line)
        issues = collect_issues(row)
        errors = [issue.message for issue in issues if issue.severity == "ERROR"]
        if errors:
            raise ValueError("复核导出遇到 ERROR：" + "；".join(errors))
        rev = [i for i in issues if i.severity == "REVIEW"]
        if not rev:
            continue
        out = copy.deepcopy(row)
        out["_manual_review"] = [{"code": i.code, "message": i.message} for i in rev]
        out_rows.append(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as output:
        output.write("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n")
    return len(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=ROOT / "experiments" / "routing_data" / "outputs" / "ft_qa_500_train.jsonl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="清洗后 JSONL（默认与输入同目录 ft_qa_500_train.cleaned.jsonl）",
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Markdown 报告路径",
    )
    ap.add_argument(
        "--extract-review",
        action="store_true",
        help="仅导出需人工复核（REVIEW）样本到 *.manual_review.jsonl",
    )
    ap.add_argument(
        "--review-out",
        type=Path,
        default=None,
        help="人工复核子集输出路径（默认 与输入同目录 *.manual_review.jsonl）",
    )
    ap.add_argument("--dry-run", action="store_true", help="不写 cleaned 文件，只写报告")
    args = ap.parse_args()

    # 转成绝对路径：传相对路径时，写报告里的 relative_to(ROOT) 会报错
    inp = args.input.resolve()
    if not inp.is_file():
        raise SystemExit(f"missing {inp}")

    out_dir = inp.parent
    if args.extract_review:
        review_path = args.review_out or (out_dir / f"{inp.stem}.manual_review.jsonl")
        n = extract_manual_review(inp, review_path)
        print(f"人工复核子集: {n} 条 -> {review_path}")
        return

    out_jsonl = None if args.dry_run else (args.out or (out_dir / f"{inp.stem}.cleaned.jsonl"))
    report = args.report or (out_dir / f"{inp.stem}.CLEAN_REPORT.md")

    run(inp, out_jsonl=out_jsonl, out_report=report, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
