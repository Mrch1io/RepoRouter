# src/agent/tools/search_tool.py
import json
from typing import Optional

import time
from functools import wraps
from threading import Lock

from src.rag.vectorstore import VectorStore
from src.rag.query_rewriter import Query_rewriter
from src.config import settings

_vector_store: Optional[VectorStore] = None
_vector_store_lock = Lock()
_rewriter: Optional[Query_rewriter] = None

def time_logger(func):
    @wraps(func) # 保持原函数的元信息（名字、注释等）不变
    def wrapper(*args, **kwargs):
       
        # perf_counter 是专门为了测算极短时间间隔设计的，精度极高，且不受系统时钟回拨影响。
        start_time = time.perf_counter() 
        
        # 执行原本的函数逻辑（无论是工具调用还是聊天主循环）
        result = func(*args, **kwargs) 
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        #打印耗时日志
        print(f"\n  函数 '{func.__name__}' 执行完毕 | 耗时: {elapsed_time:.3f} 秒\n")
        
        return result
    return wrapper

def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        with _vector_store_lock:
            if _vector_store is None:
                print("[System] 首次加载向量库...")
                store = VectorStore()
                store.load()
                _vector_store = store
    return _vector_store


def _get_rewriter() -> Query_rewriter:
    global _rewriter
    if _rewriter is None:
        print("[System] 首次加载 Query Rewriter...")
        _rewriter = Query_rewriter()
    return _rewriter

@time_logger
def search_codebase(query: str, top_k: int = 6) -> str:
    """
    Tool: 在本地代码库检索，返回一个 JSON 对象字符串，字段有：
    query、rewritten_query、top1_distance、top1_source、soft_fuse_triggered、results。
    触发软熔断时还有 soft_fuse_message。
    results 每项有 source、content（片段前 900 个字符），按语法树切块的索引还有 lines。
    """
    vs = _get_vector_store()

    clean_query = query
    if getattr(settings, "enable_query_rewrite_in_tool", False):
        try:
            rewriter = _get_rewriter()
            clean_query = rewriter.rewrite(query).dense_query or query
            print(f"[Rewrite] '{query}' -> '{clean_query}'")
        except Exception as e:
            print(f"[Rewrite Failed] fallback. err={e}")
            clean_query = query

    fuse_on = getattr(settings, "soft_fuse_enabled", False)
    thr = getattr(settings, "soft_fuse_distance_threshold", None)
    docs, doc_top1, dist_top1 = vs.search_dense_with_top1(
        clean_query,
        top_k=top_k,
        max_distance=thr if fuse_on else None,
    )

    payload: dict = {
        "query": query,
        "rewritten_query": clean_query,
        "top1_distance": dist_top1,
        "top1_source": str(doc_top1.metadata.get("source", "")) if doc_top1 else None,
        "soft_fuse_triggered": False,
        "results": [],
    }

    if (
        fuse_on
        and thr is not None
        and dist_top1 is not None
        and dist_top1 > thr
    ):
        payload["soft_fuse_triggered"] = True
        payload["soft_fuse_message"] = (
            "检索置信度不足（向量距离过高），已跳过注入代码片段；请直接基于通用知识回答或向用户说明无法从当前代码库定位。"
        )
        return json.dumps(payload, ensure_ascii=False)

    results = []
    for d in docs:
        item = {
            "source": str(d.metadata.get("source", "")),
            "content": d.page_content[:900],
        }
        #按语法树切块的库带行号，模型回答时可以直接说“第几行”
        if "start_line" in d.metadata:
            item["lines"] = f"{d.metadata['start_line']}-{d.metadata['end_line']}"
        results.append(item)
    payload["results"] = results

    return json.dumps(payload, ensure_ascii=False)
