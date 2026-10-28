#把 E2E 的 90 条人工题转成 Phase 2A 路由测试集（ShareGPT 格式）
#为什么要做：原来的 50 条测试题和训练题都是同一批 Gemini 生成的（同分布），分数偏乐观
#这 90 条是人工写的、训练时没见过，用来考“没见过的题”上派单准不准
#
#用法（项目根目录）：python -m loRA.build_cross_repo_eval
#输出：loRA/eval_cross_repo.json，可以直接给 eval_routing_*.py 的 --test-data
#有简介时复用历史 87 题的完整 system prompt 快照，和运行时 router.py 一致
#  加 --no-repo-summary 生成旧版（无简介）：loRA/eval_cross_repo_no_summary.json
#
#局限：这批题只考“该不该查代码库”（search_codebase / no_tool），不考其他 5 个工具和 clarify
import json
import sys
from pathlib import Path

from loRA.format_to_sharegpt import routing_system_prompt,repoagent_routing_system_prompt
from src.evaluation.e2e_manifest import build_e2e_manifest

NO_SUMMARY = "--no-repo-summary" in sys.argv
OUT_PATH = Path(__file__).resolve().parent / ("eval_cross_repo_no_summary.json" if NO_SUMMARY else "eval_cross_repo.json")


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
    system = routing_system_prompt(None) if NO_SUMMARY else repoagent_routing_system_prompt()
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

    OUT_PATH.write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding="utf-8")
    n_tool = sum(1 for c in cases if '"tool_call"' in c["conversations"][2]["value"])
    print(f"写入 {len(cases)} 条到 {OUT_PATH}（tool_call {n_tool} 条，no_tool {len(cases) - n_tool} 条）")
    print(f"跳过 optional 题：{skipped}")
