#廉价路由基线：训练集定多数类和规则，测试集各评一次，四个k共用嵌入
#用法：python -m lora.eval_routing_rules --train data/v2/ablation_v2_no_summary/data/routing_train.json
import argparse
from collections import Counter
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import time

from lora.eval_routing_api import Phase2aMetrics,read_gold_answer
from lora.eval_routing_knn import _normalize,fill_args,knn_predict
from src.evaluation.stats import wilson_ci


#只看637条训练题后固定，不根据测试结果改词或挑k
KEYWORDS = ("项目里","仓库","我们","这个函数","哪个文件","代码库","库里","这个接口")
IDENTIFIER = re.compile(
    r"[A-Za-z][A-Za-z0-9_]*_[A-Za-z0-9_]*|_[A-Za-z0-9_]+|[a-z][A-Z]|[A-Z]{2,}[a-z]|"
    r"[A-Za-z0-9_]+\.py(?![A-Za-z0-9_])|[A-Za-z_][A-Za-z0-9_.]*\s*[(（]"
)
KS = (5,1,3,10)
TEST_DATA = (
    "lora/eval_cross_repo_no_summary.json",
    "lora/eval_in_domain_no_summary.json",
    "data/v2/newrepo_nanochat60/eval_nanochat60_no_summary.json",
)


def load_data(path : str)->tuple:
    #1.保留输入指纹；坏金标直接停，避免偷偷改变分母
    raw = Path(path).read_bytes()
    cases = json.loads(raw)
    golds = [read_gold_answer(case) for case in cases]
    if not cases or any(gold is None for gold in golds):
        raise ValueError(f"数据为空或有坏金标：{path}")
    #2.只取问题，不把system里的信息送进规则或嵌入
    questions = [case["conversations"][1]["value"] for case in cases]
    if any(not isinstance(q,str) or not q.strip() for q in questions):
        raise ValueError(f"问题为空或不是文本：{path}")
    return questions,golds,hashlib.sha256(raw).hexdigest()


def rules_predict(question : str)->dict:
    #1.命中本仓库指示词或代码标识符，就搜索原问题
    call = any(word in question for word in KEYWORDS) or bool(IDENTIFIER.search(question))
    #2.复用近邻基线的参数填充，不输出clarify_needed
    return fill_args({"decision_type":"tool_call" if call else "no_tool",
                      "tool_name":"search_codebase" if call else None},question)


def score(golds : list,preds : list)->dict:
    #1.每题只更新一次，沿用Phase2aMetrics的三分类和调用口径
    if len(golds) != len(preds):
        raise ValueError("预测条数与金标不一致")
    metrics = Phase2aMetrics(n_file_rows=len(golds))
    for gold,pred in zip(golds,preds):
        metrics.update(gold,pred)
    #2.保存整数分子分母和区间，避免只剩四舍五入后的百分数
    result = {}
    for name,hits,total in (
        ("decision_accuracy",metrics.n_decision_match,metrics.n_evaluated),
        ("relevance",metrics.n_relevance_ok,metrics.n_gold_tool_call),
        ("irrelevance",metrics.n_irrelevance_ok,metrics.n_gold_no_call),
    ):
        result[name] = {"correct":hits,"total":total,"rate":hits / total if total else None}
    result["decision_accuracy"]["wilson95"] = wilson_ci(metrics.n_decision_match,metrics.n_evaluated)
    return result


def write_json(path : Path,value : dict)->None:
    #1.独占创建，已有结果绝不覆盖
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")


def embed_questions(emb,questions : list,label : str)->tuple:
    #1.每批最多10条，和现有Qwen配置一致；计的是文本条数和库函数调用数
    start = time.perf_counter()
    vectors = []
    calls = 0
    for offset in range(0,len(questions),10):
        batch = questions[offset:offset + 10]
        raw = emb.embed_documents(batch)
        if len(raw) != len(batch) or any(not v or not any(v) for v in raw):
            raise ValueError(f"嵌入返回空向量或条数错误：{label}")
        vectors.extend(_normalize(v) for v in raw)
        calls += 1
        if calls % 10 == 0 or len(vectors) == len(questions):
            logging.info("嵌入 %s：%s/%s条",label,len(vectors),len(questions))
    #2.校验维度，防止原近邻实现的zip静默截断
    dimensions = {len(v) for v in vectors}
    if len(dimensions) != 1:
        raise ValueError(f"嵌入维度不一致：{label}")
    return vectors,{"texts":len(questions),"embed_documents_calls":calls,
                    "dimensions":len(vectors[0]),"seconds":time.perf_counter() - start}


def save_result(path : Path,golds : list,preds : list,seconds : float)->dict:
    #1.记录预测，之后核对结果时不用重新跑测试集
    result = {"metrics":score(golds,preds),"prediction_seconds":seconds,"predictions":[
        {"index":i,"gold":gold,"prediction":pred} for i,(gold,pred) in enumerate(zip(golds,preds))]}
    #2.每个基线和测试集各一个文件，运行汇总只记索引和耗时
    write_json(path,result)
    accuracy = result["metrics"]["decision_accuracy"]
    logging.info("%s：%s/%s，预测 %.4f秒",path.stem,accuracy["correct"],accuracy["total"],seconds)
    return {"file":path.name,"prediction_seconds":seconds}


def main()->None:
    #1.已有输出目录就停；先固定训练信息，再加载测试题
    parser = argparse.ArgumentParser(description="廉价路由基线：多数类、关键词、近邻")
    parser.add_argument("--train",default="data/v2/ablation_v2_no_summary/data/routing_train.json")
    parser.add_argument("--test-data",nargs="+",default=list(TEST_DATA))
    args = parser.parse_args()
    output = Path("data/v2/cheap_baselines")
    if output.exists():
        parser.error(f"目标已存在，停止以保留结果：{output}")
    if len({Path(p).stem for p in args.test_data}) != len(args.test_data):
        parser.error("测试集文件名重复，会造成结果冲突")
    start = time.perf_counter()
    train_q,train_g,train_hash = load_data(args.train)
    counts = Counter(g["decision_type"] for g in train_g)
    majority = counts.most_common(1)[0][0]
    majority_tool = Counter(g.get("tool_name") for g in train_g
                            if g["decision_type"] == majority).most_common(1)[0][0]
    train_info = {"path":args.train,"sha256":train_hash,"rows":len(train_g),
                  "decision_counts":dict(counts),"majority_decision":majority}
    rule_info = {"keywords":KEYWORDS,"identifier_regex":IDENTIFIER.pattern,
                 "tuning_data":args.train,"train_metrics":score(train_g,[rules_predict(q) for q in train_q])}
    output.mkdir()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",handlers=[
        logging.FileHandler(output / "run.log",mode="x",encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ])
    logging.info("训练集%s条，类别%s；规则冻结，主结果k=5，其余只作敏感性参考",len(train_g),dict(counts))
    run = {"train":train_info,"rules":rule_info,"primary_k":5,"sensitivity_k":[1,3,10],
           "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"test_sha256":{},"results":[]}
    write_json(output / "protocol.json",run)

    #2.同现有近邻脚本用同一嵌入配置，训练向量只算一次
    from src.rag.vectorstore import VectorStore
    emb = VectorStore().embeddings
    run["embedding_model"] = emb.model
    train_vecs,train_cost = embed_questions(emb,train_q,"train")
    run["embedding_train"] = train_cost
    run["embedding_tests"] = {}
    for test_path in args.test_data:
        questions,golds,test_hash = load_data(test_path)
        run["test_sha256"][test_path] = test_hash
        #3.常量和规则不调用API；多数类只由训练金标决定
        for name in ("always_no_tool","majority","keywords"):
            tick = time.perf_counter()
            if name == "keywords":
                preds = [rules_predict(q) for q in questions]
            else:
                decision = "no_tool" if name == "always_no_tool" else majority
                tool = majority_tool if decision == "tool_call" else None
                preds = [fill_args({"decision_type":decision,"tool_name":tool},q) for q in questions]
            seconds = time.perf_counter() - tick
            path = output / f"{name}__{Path(test_path).stem}.json"
            run["results"].append(save_result(path,golds,preds,seconds))

        #4.测试向量也只算一次，四个k原样调用已有近邻投票，不按结果选k
        test_vecs,test_cost = embed_questions(emb,questions,Path(test_path).stem)
        if test_cost["dimensions"] != train_cost["dimensions"]:
            raise ValueError("训练与测试嵌入维度不同")
        run["embedding_tests"][test_path] = test_cost
        for k in KS:
            tick = time.perf_counter()
            preds = knn_predict(test_vecs,train_vecs,train_g,k=k)
            preds = [fill_args(pred,q) for pred,q in zip(preds,questions)]
            seconds = time.perf_counter() - tick
            path = output / f"knn_k{k}__{Path(test_path).stem}.json"
            run["results"].append(save_result(path,golds,preds,seconds))

    #5.只加实际发生的嵌入，不重复累加各k引用的共享成本
    costs = [train_cost,*run["embedding_tests"].values()]
    run["embedding_totals"] = {key:sum(cost[key] for cost in costs)
                                for key in ("texts","embed_documents_calls","seconds")}
    run["prediction_seconds"] = sum(row["prediction_seconds"] for row in run["results"])
    run["total_seconds"] = time.perf_counter() - start
    run["timing_note"] = "总耗时含加载和初始化；嵌入耗时含归一化；库函数调用数不含SDK内部重试。"
    write_json(output / "run.json",run)
    logging.info("完成：嵌入%s，总耗时%.3f秒",run["embedding_totals"],run["total_seconds"])


if __name__ == "__main__":
    main()
