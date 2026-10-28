"""
重新生成的数据：修补 ft_qa_500_train.cleaned.jsonl（一次性脚本，可审计）。

- 7 条 clarify_needed → tool_call(search_codebase)：题目给了明确的文件/类/接口，先查库
- misleading_edge_0082 → tool_call(analyse_code)：题干已贴出代码
- misleading_edge_0086 → no_tool：要求 git push / 重启容器，超出工具能力，也不需要追问
- 删除 query 完全重复的样本（保留第一条）
其余 35 条复核项保持原样。输出 ft_qa_500_train.reviewed.jsonl，不改输入文件。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments" / "routing_data" / "outputs"
INP = OUT_DIR / "ft_qa_500_train.cleaned.jsonl"
OUT = OUT_DIR / "ft_qa_500_train.reviewed.jsonl"

# sample_id -> 检索词
TO_SEARCH = {
    "domain_0138": "app/api/routers/inference.py 推理路由函数 参数校验",
    "domain_0118": "app/middleware/rate_limit.py 限流 校验函数",
    "domain_0160": "app/middleware/rate_limit.py 核心限流函数",
    "domain_0119": "/v1/embed 接口 实现 异常处理",
    "domain_0374": "BatchScheduler 队列 submit _flush",
    "misleading_edge_0067": "订单状态机 状态转换函数 调用",
    "misleading_edge_0088": "登录接口 login 实现",
}

TO_ANALYSE_CODE = {
    "misleading_edge_0082": "def calc_discount(price, rate):\n    if rate > 1:\n        # TODO: 处理异常汇率与会员叠加逻辑",
}

TO_NO_TOOL = {"misleading_edge_0086"}


def main() -> None:
    #1.先保护已经复核的训练数据，失败时不改旧文件
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError(f"拒绝覆盖已复核数据：{OUT}")
    rows = [json.loads(ln) for ln in INP.read_text(encoding="utf-8").splitlines() if ln.strip()]
    seen: set[str] = set()
    out: list[dict] = []
    dup = 0
    for r in rows:
        q = r["query"].strip()
        if q in seen:
            dup += 1
            continue
        seen.add(q)
        sid = r["sample_id"]
        meta = dict(r.get("_meta") or {})
        if sid in TO_SEARCH:
            r.update(decision_type="tool_call", tool_name="search_codebase",
                     arguments={"query": TO_SEARCH[sid], "top_k": 6}, clarify_answer=None, oracle_context=None)
            meta["manual_review"] = "clarify_needed → search_codebase"
        elif sid in TO_ANALYSE_CODE:
            r.update(decision_type="tool_call", tool_name="analyse_code",
                     arguments={"code": TO_ANALYSE_CODE[sid]}, clarify_answer=None, oracle_context=None)
            meta["manual_review"] = "clarify_needed → analyse_code"
        elif sid in TO_NO_TOOL:
            r.update(decision_type="no_tool", tool_name=None, arguments=None, clarify_answer=None, oracle_context=None)
            meta["manual_review"] = "clarify_needed → no_tool"
        r["_meta"] = meta
        out.append(r)

    changed = sum(1 for r in out if "manual_review" in (r.get("_meta") or {}))
    assert changed == len(TO_SEARCH) + len(TO_ANALYSE_CODE) + len(TO_NO_TOOL), changed
    with OUT.open("x", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n")
    print(f"读入 {len(rows)} 条，删除重复 {dup} 条，修改 {changed} 条 -> {OUT.name}（{len(out)} 条）")


if __name__ == "__main__":
    main()
