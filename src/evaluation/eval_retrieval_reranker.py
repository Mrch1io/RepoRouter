from src.evaluation.rag_dataset import EVAL_QUERIES
from src.evaluation.stats import hit_and_rr
from src.rag.vectorstore import VectorStore
import time

#量化评估脚本

#main准备：构建向量数据库，不同模式的量化指标字典推导式
if __name__ == "__main__":


    vs = VectorStore(use_reranker=True)
    vs.load()

    modes = {"dense","bm25","hybrid"}

    #为每种场景分别计分
    from collections import defaultdict
    #自动初始化字典数据结构：使用匿名函数接受传入的键。如果字典中没有，新构建一个
    stats = defaultdict(lambda : {m:{"hit":0,"rr":0.0,"count":0} for m in modes})



    #全局的计分
    global_stats = {m:{"hit":0,"rr":0.0} for m in modes} #每种模式构建一个存值字典。

    top_k = 8
    #基准测试

    #提取问题，提取答案，算出预期，进行量化

    for case in EVAL_QUERIES:
        q = case["query"]
        exp = case["expected_files"]
        #获取问题类型
        q_type = case["query_type"]
        

        res_dense = vs.search_dense(q,top_k = top_k)
        time.sleep(1.5)
        res_bm25 = vs.search_bm25(q,top_k = top_k)
        time.sleep(1.5)
        res_hybrid = vs.search_hybrid(q,top_k = top_k)

        result = {
            "dense" : res_dense,
            "bm25" : res_bm25,
            "hybrid" : res_hybrid
        }


        #for key,values in dict.items() :返回每个键值对的元组。如果不加items()，遍历字典只返回键
        for mode,docs in result.items():
            hit,rr = hit_and_rr(docs,exp)
            #把量化数据加进去
            stats[q_type][mode]["hit"] += hit 
            stats[q_type][mode]["rr"] += rr
            #更新数量
            stats[q_type][mode]["count"] += 1

            #把量化数据加进去
            global_stats[mode]["hit"] += hit 
            global_stats[mode]["rr"] += rr

    #不同提问类型的结果
    for q_type,type_stats in stats.items():
        count = type_stats["dense"]["count"]

        print(f"======================== 提问类型:{q_type} =======================\n")
        print(f"======================== 问题数量:{count} =======================\n")

        for mode in modes:          #字符串键：加双引号。
            hit_rate = type_stats[mode]["hit"] / count
            mrr = type_stats[mode]["rr"] / count
        
            print(f"{q_type}类型提问：引擎：{mode} | Hit_Rate：{hit_rate*100:.1f}% | MRR:{mrr:.3f}") 





    #计算最终平均结果

    n = len(EVAL_QUERIES) #总题数

    print("="*100)
    print("=="*25,f"最终全局评测结果(top_k = {top_k})","=="*25)

    for mode in modes:          #字符串键：加双引号。
        hit_rate = global_stats[mode]["hit"] / n
        mrr = global_stats[mode]["rr"] / n
        
        print(f"引擎：{mode} | Hit_Rate：{hit_rate*100:.1f}% | MRR:{mrr:.3f}") 
