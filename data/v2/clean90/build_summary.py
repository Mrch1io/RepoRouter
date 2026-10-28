import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "data/v2/clean90"
FROZEN = ROOT / "data/v2/frozen90"
COLUMNS = (("frozen_native", "frozen 原生", "frozen", "native", "A"),
           ("frozen_q4", "frozen Q4", "frozen", "q4", "B"),
           ("clean_native", "clean 原生", "clean", "native", "A"),
           ("clean_q4", "clean Q4", "clean", "q4", "B"))
ROWS = (("route_correct", "路由正确 /90"),
        ("must_retrieve_actual", "35 道必须检索题实际检索数"),
        ("judge_correct", "自动评委判正确 /90"),
        ("remaining68_correct", "剩余 68 题判正确"),
        ("score", "平均评分"), ("seconds", "平均耗时（秒）"),
        ("cost_usd", "每题 DeepSeek 费用（美元）"),
        ("answer_chars", "平均回答字数（字符）"),
        ("usage_calls", "DeepSeek 调用次数/题"))


def read_json(path : Path)->dict:
    return json.loads(path.read_text("utf-8"))


def read_answers(path : Path)->dict:
    #1.只读完整回答，题号不能重复
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    answers = {row["sample_id"]: row for row in rows}
    assert len(rows) == len(answers) == 90, str(path)
    return answers


def column_values(comparison : dict, side : str)->dict:
    #1.直接取冻结比较器的聚合结果，不另算均值或检验
    all_rows = comparison["subsets"]["all"]
    remaining = comparison["subsets"]["remaining68"]
    assert all_rows["n"] == 90 and all_rows["must_ground_n"] == 35 and remaining["n"] == 68
    return {
        "route_correct": all_rows["binary"]["l1_correct"][side + "_correct"],
        "must_retrieve_actual": all_rows["must_ground_retrieval"][side]["searched"],
        "judge_correct": all_rows["binary"]["final_judgement"][side + "_correct"],
        "remaining68_correct": remaining["binary"]["final_judgement"][side + "_correct"],
        **{field: all_rows["metrics"][field][side + "_mean"]
           for field in ("score", "seconds", "cost_usd", "answer_chars", "usage_calls")},
    }


def format_number(value, field : str = "")->str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.9f}" if field == "cost_usd" else f"{value:.6f}"


def main():
    #1.先检查输出和输入，已有文件不覆盖
    output, report_path = EXP / "summary.json", EXP / "结果.md"
    assert not output.exists() and not report_path.exists(), "summary.json or 结果.md already exists"
    paths = {"frozen_comparison": FROZEN / "comparison.json", "clean_comparison": EXP / "comparison.json",
             "retrieval_audit": EXP / "retrieval_audit.json"}
    comparisons = {name: read_json(paths[name + "_comparison"]) for name in ("frozen", "clean")}
    audit = read_json(paths["retrieval_audit"])
    assert audit["status"] == "passed"
    assert comparisons["clean"]["review_plan"] is None
    parent = read_json(FROZEN / "run.json")
    manifest_ids = {item["id"] for item in parent["manifest"]}
    for relative, digest in parent["source_sha256_frozen"].items():
        assert hashlib.sha256((FROZEN / "source" / relative).read_bytes()).hexdigest() == digest, relative
    answers, pairs, columns = {}, {}, {}
    for name, comparison in comparisons.items():
        assert comparison["publishable"], name + " comparison contains errors"
        pairs[name] = {pair["sample_id"]: pair for pair in comparison["pairs"]}
        assert len(comparison["pairs"]) == 90 and set(pairs[name]) == manifest_ids
        directory = FROZEN if name == "frozen" else EXP
        answers[name] = {}
        for backend in ("native", "q4"):
            path = directory / f"e2e_{backend}.jsonl"
            paths[name + "_" + backend] = path
            answers[name][backend] = read_answers(path)
            assert set(answers[name][backend]) == manifest_ids
            assert hashlib.sha256(path.read_bytes()).hexdigest() == audit["experiments"][name][backend]["input_sha256"]
    for key, label, name, backend, side in COLUMNS:
        columns[key] = {"label": label, "comparison_side": side, **column_values(comparisons[name], side)}

    #2.按题配对列出变化，正确率口径仍是 correct 与非 correct
    judgement_changes, label_only_changes, routing_changes = {}, {}, {}
    for backend, side in (("native", "A"), ("q4", "B")):
        judgement_changes[backend], label_only_changes[backend], routing_changes[backend] = [], [], []
        for identity in sorted(manifest_ids):
            before = pairs["frozen"][identity][side]["final_judgement"]
            after = pairs["clean"][identity][side]["final_judgement"]
            evidence = audit["experiments"]["frozen"][backend]["samples"][identity]
            if before != after:
                change = {"sample_id": identity, "frozen_judgement": before, "clean_judgement": after,
                          "frozen_correct": before == "correct", "clean_correct": after == "correct",
                          "frozen_retrieved_exam_file": evidence["retrieved_excluded_file"],
                          "frozen_exam_files": evidence["excluded_files"]}
                destination = judgement_changes if (before == "correct") != (after == "correct") else label_only_changes
                destination[backend].append(change)
            old, new = answers["frozen"][backend][identity], answers["clean"][backend][identity]
            if backend == "native":
                old_decision, new_decision = old["actual_call_search_codebase"], new["actual_call_search_codebase"]
            else:
                old_decision = old["router_decision"]["decision_type"]
                new_decision = new["router_decision"]["decision_type"]
            if old_decision != new_decision:
                routing_changes[backend].append({"sample_id": identity, "frozen": old_decision, "clean": new_decision})

    #3.置信区间和检验直接引用 clean 比较结果，保留 Q4 减原生方向
    clean_all = comparisons["clean"]["subsets"]["all"]
    paired = {"direction": comparisons["clean"]["delta_direction"],
              "mcnemar": {field: clean_all["binary"][field] for field in ("final_judgement", "l1_correct")},
              "differences": {field: clean_all["metrics"][field] for field in ("score", "seconds", "cost_usd")}}
    assert paired["direction"] == "B_minus_A"
    audit_fields = ("retrieval_calls", "returned_fragments", "evaluation_retrieval_calls", "excluded_retrieval_calls",
                    "question_verbatim_retrieval_calls", "evaluation_source_retrieval_counts", "excluded_source_retrieval_counts")
    audit_summary = {key: {field: audit["experiments"][name][backend][field] for field in audit_fields}
                     for key, label, name, backend, side in COLUMNS}
    misses = {identity: {"actual_call_search_codebase": answers["clean"]["q4"][identity]["actual_call_search_codebase"],
                        "decision_type": answers["clean"]["q4"][identity]["router_decision"]["decision_type"]}
              for identity in ("p04", "p18", "p29")}
    notes = [
        "表中聚合统计直接取两份 comparison.json；McNemar 和 bootstrap 95% 区间直接取 clean comparison.json，未重新计算。",
        "剩余 68 题排除冻结代码定义的 known_pilot22；自动判正确仅指 final_judgement=correct，partially_correct 与 incorrect 均为非正确。",
        "平均回答字数沿用冻结代码 len(final_answer)，包括标点与代码字符；费用按冻结价格表估算全部 DeepSeek 调用，不含嵌入与 GPU 租金。",
        "原生链路先决定查不查，再读取检索结果。Q4 独立路由温度为 0.1，原生工具决策及 DeepSeek 回答温度为 0.2；跨运行查不查变化属于采样差异，不能归因于索引。",
        "两次运行时段不同，只报告观察到的差异，不作索引变化与答案质量之间的因果结论。此次没有重做人工盲评。",
        "审计以 tool_actions 中 search_codebase 调用为单位，关联 tool_contexts 读取片段；同次同来源去重，各来源次数可重叠。完整来源分布及逐题证据见 retrieval_audit.json。",
    ]
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "columns": columns,
              "clean_paired_comparison": paired, "judgement_correctness_changes": judgement_changes,
              "judgement_label_only_changes": label_only_changes, "routing_decision_changes": routing_changes,
              "retrieval_audit": audit_summary, "clean_q4_p04_p18_p29": misses, "notes": notes,
              "inputs": {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for key, path in paths.items()}}

    #4.简短报告只排版以上结果，不引入另一套统计
    labels = [column[1] for column in COLUMNS]
    lines = ["本次汇总使用冻结比较代码的已有统计；A=原生 Flash，B=Q4 路由 + Flash。", "",
             "| 指标 | " + " | ".join(labels) + " |", "|---|---:|---:|---:|---:|"]
    for field, label in ROWS:
        lines.append("| " + label + " | " + " | ".join(format_number(columns[key][field], field) for key, *_ in COLUMNS) + " |")
    lines += ["", "clean 两链配对比较（差值均为 Q4 − 原生）：", "",
              f"- 对错 McNemar p：{paired['mcnemar']['final_judgement']['mcnemar_p']}。",
              f"- 路由 McNemar p：{paired['mcnemar']['l1_correct']['mcnemar_p']}。"]
    for field, label in (("score", "评分"), ("seconds", "耗时（秒）"), ("cost_usd", "每题费用（美元）")):
        metric = paired["differences"][field]
        lines.append(f"- {label}差：{metric['mean_delta_B_minus_A']}；bootstrap 95% 区间：{metric['delta_ci95']}。")
    for title, changes in (("判对/判错变化", judgement_changes), ("仅三分类标签变化，正确布尔值不变", label_only_changes)):
        lines += ["", title + "（frozen → clean；考题命中指 frozen 时实际检索）：", ""]
        for backend, items in changes.items():
            lines.append(f"- {backend}：" + ("；".join(f"{item['sample_id']} {item['frozen_judgement']} → {item['clean_judgement']}，考题命中{'是' if item['frozen_retrieved_exam_file'] else '否'}" for item in items) or "无"))
    lines += ["", "路由决策变化（与判分变化分开）：", ""]
    for backend, items in routing_changes.items():
        lines.append(f"- {backend}：" + ("；".join(f"{item['sample_id']} {item['frozen']} → {item['clean']}" for item in items) or "无"))
    lines += ["", "clean Q4 的 p04、p18、p29：" + "；".join(f"{identity} {'仍漏查' if not value['actual_call_search_codebase'] else '已检索'}（{value['decision_type']}）" for identity, value in misses.items()) + "。", "",
              "| 检索审计 | " + " | ".join(labels) + " |", "|---|---:|---:|---:|---:|"]
    for field, label in (("retrieval_calls", "检索次数"), ("returned_fragments", "返回片段数"),
                         ("evaluation_retrieval_calls", "命中任意 evaluation 文件"),
                         ("excluded_retrieval_calls", "命中四个排除文件"),
                         ("question_verbatim_retrieval_calls", "片段包含当前问题原文")):
        lines.append("| " + label + " | " + " | ".join(str(audit_summary[key][field]) for key, *_ in COLUMNS) + " |")
    evaluation_sources = sorted({source for entry in audit_summary.values() for source in entry["evaluation_source_retrieval_counts"]})
    lines += ["", "| evaluation 来源文件（命中检索次数） | " + " | ".join(labels) + " |", "|---|---:|---:|---:|---:|"]
    for source in evaluation_sources:
        lines.append("| " + source + " | " + " | ".join(str(audit_summary[key]["evaluation_source_retrieval_counts"].get(source, 0)) for key, *_ in COLUMNS) + " |")
    lines += ["", *["- " + note for note in notes], ""]
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print("Summary created:", output, report_path)


if __name__ == "__main__":
    main()
