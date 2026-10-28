from src.evaluation.rag_dataset import EVAL_QUERIES
from src.evaluation.stats import wilson_ci,bootstrap_ci,hit_and_rr
from src.config import settings
from src.rag.vectorstore import VectorStore

import argparse
import json
from pathlib import Path

#量化评估脚本
#用法（项目根目录，先用 scripts/build_knowledge_base.py 建好对应的库）：
#  python -m src.evaluation.eval_retrieval
#  python -m src.evaluation.eval_retrieval --embedding qwen3.7-text-embedding-flash --persist-dir data/indices/src_flash
#  python -m src.evaluation.eval_retrieval --rerank --json-out data/v2/retrieval_rerank.json

#main准备：构建向量数据库，不同模式的量化指标字典推导式
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检索评估：dense / bm25 / hybrid")
    parser.add_argument("--persist-dir",default=settings.Chroma_persist_dir)
    parser.add_argument("--embedding",default=settings.qwen_embedding_name)
    parser.add_argument("--rerank",action="store_true",help="打开云端 qwen 重排")
    parser.add_argument("--top-k",type=int,default=8)
    parser.add_argument("--json-out",default=None)
    args = parser.parse_args()

    #1.已有结果先拒绝，避免跑完付费评测才发现目标不能写
    output = Path(args.json_out) if args.json_out else None
    if output is not None and (output.exists() or output.is_symlink()):
        parser.error(f"输出文件已存在，请用 --json-out 指定新文件：{output}")

    #改配置：要和建库时用的 embedding、目录一致
    settings.Chroma_persist_dir = args.persist_dir
    settings.bm25_persist_dir = str(Path(args.persist_dir) / "bm25_index.pkl")
    settings.qwen_embedding_name = args.embedding

    vs = VectorStore(use_reranker=args.rerank)
    vs.load()

    modes = ["dense","bm25","hybrid"]   #用列表，打印顺序固定

    #为每种场景分别计分
    from collections import defaultdict
    #自动初始化字典数据结构：使用匿名函数接受传入的键。如果字典中没有，新构建一个
    stats = defaultdict(lambda : {m:{"hit":0,"rr":0.0,"count":0} for m in modes})

    #全局的计分：每题的 hit / rr 都留下来，后面算置信区间
    per_query = {m:{"hit":[],"rr":[]} for m in modes}

    top_k = args.top_k
    #基准测试

    #提取问题，提取答案，算出预期，进行量化

    for case in EVAL_QUERIES:
        q = case["query"]
        exp = case["expected_files"]
        #获取问题类型
        q_type = case["query_type"]

        result = {
            "dense" : vs.search_dense(q,top_k = top_k),
            "bm25" : vs.search_bm25(q,top_k = top_k),
            "hybrid" : vs.search_hybrid(q,top_k = top_k)
        }


        #for key,values in dict.items() :返回每个键值对的元组。如果不加items()，遍历字典只返回键
        for mode,docs in result.items():
            hit,rr = hit_and_rr(docs,exp)
            #把量化数据加进去
            stats[q_type][mode]["hit"] += hit
            stats[q_type][mode]["rr"] += rr
            #更新数量
            stats[q_type][mode]["count"] += 1
            per_query[mode]["hit"].append(hit)
            per_query[mode]["rr"].append(rr)

    #不同提问类型的结果
    for q_type,type_stats in stats.items():
        count = type_stats["dense"]["count"]

        print(f"======================== 提问类型:{q_type} =======================\n")
        print(f"======================== 问题数量:{count} =======================\n")

        for mode in modes:          #字符串键：加双引号。
            hit_rate = type_stats[mode]["hit"] / count
            mrr = type_stats[mode]["rr"] / count

            print(f"{q_type}类型提问：引擎：{mode} | Hit_Rate：{hit_rate*100:.1f}% | MRR:{mrr:.3f}")


    #计算最终平均结果（带95%置信区间：只有18题，区间会比较宽）

    n = len(EVAL_QUERIES) #总题数

    print("="*100)
    print("=="*25,f"最终全局评测结果(top_k = {top_k}，embedding = {args.embedding}，rerank = {args.rerank})","=="*25)

    #persist_dir 里能看出是哪种切块（比如 src_ast）
    summary = {"embedding":args.embedding,"persist_dir":args.persist_dir,"rerank":args.rerank,"top_k":top_k,"n":n,"modes":{}}
    for mode in modes:          #字符串键：加双引号。
        k = sum(per_query[mode]["hit"])
        mrr = sum(per_query[mode]["rr"]) / n
        h_lo,h_hi = wilson_ci(k,n)
        m_lo,m_hi = bootstrap_ci(per_query[mode]["rr"])
        print(f"引擎：{mode} | Hit_Rate：{k/n*100:.1f}% [{h_lo*100:.1f}, {h_hi*100:.1f}] | MRR:{mrr:.3f} [{m_lo:.3f}, {m_hi:.3f}]")
        summary["modes"][mode] = {"hit_rate":k/n,"hit_ci":[h_lo,h_hi],"mrr":mrr,"mrr_ci":[m_lo,m_hi],"rr":per_query[mode]["rr"]}

    if output is not None:
        #2.独占创建，运行途中有人抢先写入也不能覆盖
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("x",encoding="utf-8") as fp:
            fp.write(json.dumps(summary,ensure_ascii=False,indent=2))
        print(f"已写入 {output}")
