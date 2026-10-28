#把 E2E 的 90 条人工题转成 Phase 2A 路由测试集（ShareGPT 格式）
#为什么要做：原来的 50 条测试题和训练题都是同一批 Gemini 生成的（同分布），分数偏乐观
#这 90 条是人工写的、训练时没见过，用来考“没见过的题”上派单准不准
#
#用法（项目根目录）：python -m lora.build_cross_repo_eval
#输出：lora/eval_cross_repo.json，可以直接给 eval_routing_*.py 的 --test-data
#有简介时复用历史 87 题的完整 system prompt 快照，和运行时 router.py 一致
#  加 --no-repo-summary 生成旧版（无简介）：lora/eval_cross_repo_no_summary.json
#
#局限：这批题只考“该不该查代码库”（search_codebase / no_tool），不考其他 5 个工具和 clarify
import json
import argparse
from pathlib import Path

from lora.format_to_sharegpt import routing_system_prompt,load_prompt_snapshot
from src.evaluation.e2e_manifest import build_e2e_manifest

#1.根据manifest的标签生成金标路由JSON
def gold_for(row : dict)->dict | None:
    if row["should_call_tool"] == "yes":
        return {"decision_type":"tool_call","tool_name":"search_codebase",
                "arguments":{"query":row["query"],"top_k":6},"clarify_answer":None,"oracle_context":None}
    if row["should_call_tool"] == "no":
        return {"decision_type":"no_tool","tool_name":None,"arguments":None,"clarify_answer":None,"oracle_context":None}
    #optional（调不调都对）的题评测脚本没法打分，先跳过
    return None


if __name__ == "__main__":
    #1.历史题集默认保留，需要重建时用--out指定新文件
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-repo-summary", action="store_true")
    ap.add_argument("--out", type=Path, help="输出到新文件，拒绝覆盖已有题集")
    args = ap.parse_args()
    out_path = args.out or Path(__file__).resolve().with_name("eval_cross_repo_no_summary.json" if args.no_repo_summary else "eval_cross_repo.json")
    if out_path.exists() or out_path.is_symlink():
        ap.error(f"拒绝覆盖已有题集，请用 --out 指定新文件：{out_path}")
    system = routing_system_prompt(None) if args.no_repo_summary else load_prompt_snapshot()
    cases = []
    skipped = []
    for row in build_e2e_manifest():
        gold = gold_for(row)
        if gold is None:
            skipped.append(row["id"])
            continue
        #2.拼成和训练数据一样的三段式对话
        cases.append({
            "id":row["id"],
            "bucket":row["bucket_type"],
            "conversations":[
                {"from":"system","value":system},
                {"from":"user","value":row["query"]},
                {"from":"assistant","value":json.dumps(gold,ensure_ascii=False)},
            ],
        })

    out_path.parent.mkdir(parents=True,exist_ok=True)
    with out_path.open("x",encoding="utf-8") as f:
        json.dump(cases,f,ensure_ascii=False,indent=2)
    n_tool = sum(1 for c in cases if '"tool_call"' in c["conversations"][2]["value"])
    print(f"写入 {len(cases)} 条到 {out_path}（tool_call {n_tool} 条，no_tool {len(cases) - n_tool} 条）")
    print(f"跳过 optional 题：{skipped}")
