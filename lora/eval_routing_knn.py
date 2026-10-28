#向量近邻路由基线（借鉴 aurelio-labs/semantic-router 的思路）
#不训练任何模型：把训练集里的问题当“例题”，新问题找最像的 k 道例题，按相似度加权投票决定派单
#用来回答“为什么要微调”：LoRA 小模型必须明显比这个几乎不花钱的方法好，微调才值得
#
#用法（项目根目录，需要 .env 里的 qwen_api_key 算 embedding）：
#  python -m lora.eval_routing_knn --train lora/routing_train.json --test-data lora/eval_cross_repo.json
#  python -m lora.eval_routing_knn --test-data lora/eval_cross_repo.json --loo
#    --loo：留一法，例题就是测试集自己（每题投票时排除它自己）。还没找回训练集时先用它，只能当参考
#
#局限：近邻法只决定“调不调、调哪个工具”，不会生成参数；参数直接用原问题填进第一个必填参数
import argparse
import math
import time
from collections import defaultdict
import json

from lora.eval_routing_api import Phase2aMetrics,read_gold_answer
from src.agent.tools.schemas import TOOLS_SCHEMA


#1.向量归一化：之后点积就是余弦相似度
def _normalize(v : list)->list:
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x / n for x in v]


#2.核心：找最像的 k 道例题，按相似度加权投票。skip_self=True 时跳过和自己同序号的例题（留一法）
def knn_predict(test_vecs : list,train_vecs : list,train_golds : list,k : int = 5,skip_self : bool = False)->list:
    preds = []
    for i,q in enumerate(test_vecs):
        sims = [(sum(a*b for a,b in zip(q,t)),j) for j,t in enumerate(train_vecs) if not (skip_self and j == i)]
        votes = defaultdict(float)
        for s,j in sorted(sims,reverse=True)[:k]:
            g = train_golds[j]
            votes[(g["decision_type"],g.get("tool_name"))] += s
        dt,tool = max(votes,key=votes.get)
        preds.append({"decision_type":dt,"tool_name":tool})
    return preds


#3.补参数：把原问题填进这个工具的第一个必填参数（search_codebase 就是 query）
def fill_args(pred : dict,question : str)->dict:
    if pred["decision_type"] == "tool_call":
        schema = next(t["function"] for t in TOOLS_SCHEMA if t["function"]["name"] == pred["tool_name"])
        pred["arguments"] = {schema["parameters"]["required"][0]:question}
    else:
        pred["arguments"] = None
        pred["clarify_answer"] = "请补充更多信息。" if pred["decision_type"] == "clarify_needed" else None
    return pred


def _load(path : str)->tuple:
    cases = json.load(open(path,encoding="utf-8"))
    pairs = [(c["conversations"][1]["value"],read_gold_answer(c)) for c in cases]
    return [q for q,g in pairs if g],[g for q,g in pairs if g]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phase 2A 路由测评 · 向量近邻基线")
    ap.add_argument("--train",default="lora/routing_train.json",help="例题（和 LoRA 用同一份训练集）")
    ap.add_argument("--test-data",default="lora/eval_cross_repo.json")
    ap.add_argument("--k",type=int,default=5)
    ap.add_argument("--loo",action="store_true",help="留一法：例题 = 测试集自己")
    args = ap.parse_args()

    #embedding 和建库用的是同一个模型（配置见 src/config.py）
    from src.rag.vectorstore import VectorStore
    emb = VectorStore().embeddings

    test_q,test_g = _load(args.test_data)
    train_q,train_g = (test_q,test_g) if args.loo else _load(args.train)

    start = time.perf_counter()
    train_vecs = [_normalize(v) for v in emb.embed_documents(train_q)]
    test_vecs = train_vecs if args.loo else [_normalize(v) for v in emb.embed_documents(test_q)]
    preds = knn_predict(test_vecs,train_vecs,train_g,k=args.k,skip_self=args.loo)
    seconds = time.perf_counter() - start

    m = Phase2aMetrics(n_file_rows=len(test_q))
    for q,g,p in zip(test_q,test_g,preds):
        m.update(g,fill_args(p,q))

    print(f"====== RepoRouter Phase 2A · 向量近邻基线（k={args.k}，{'留一法' if args.loo else f'例题 {len(train_q)} 条'}）======\n")
    for line in m.report_lines():
        print(line)
    print(f"总耗时 {seconds:.1f} 秒（大部分是算 embedding；派单本身是毫秒级）")
