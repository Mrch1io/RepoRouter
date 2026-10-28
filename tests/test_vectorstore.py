#测试向量数据库：真实embedding建一个临时小库（要qwen key，默认跳过：pytest -m live 才跑）
import pytest
import runpy
import json
import sys
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor,TimeoutError
from threading import Event

from src.agent.tools import search_tool
from src.config import settings
from src.rag.loader import RePoLoader
from src.rag.splitter import CodeSplitter
from src.rag.vectorstore import VectorStore
from src.rag import vectorstore
from src.rag.query_rewriter import Query_rewriter
from langchain_core.documents import Document


@pytest.fixture
def db(tmp_path,monkeypatch):
    monkeypatch.setattr(settings,"Chroma_persist_dir",str(tmp_path / "chroma"))
    monkeypatch.setattr(settings,"bm25_persist_dir",str(tmp_path / "bm25.pkl"))
    docs = RePoLoader("src/rag").load()
    all_chunks = CodeSplitter().split(docs)
    db = VectorStore()
    #切分好的词块给向量数据库导入
    db.index_document(all_chunks)
    return db


@pytest.mark.live
def test_dense_and_hybrid(db):
    for docs in (db.search_dense("如何把代码块存进向量数据库",top_k=3),db.search("def index_document",top_k=3)):
        assert any("vectorstore.py" in d.metadata["source"] for d in docs)


@pytest.mark.live
def test_query_rewriter():
    raw_query = "妈妈，请帮我找一找一下切分文本的函数定义def，呜呜呜。。。。。"
    res = Query_rewriter().rewrite(raw_query)
    print(res)
    #之前每次都静默降级成原问题，现在应该真的改写
    assert res.dense_query and res.dense_query != raw_query


@pytest.mark.parametrize("response",[
    None,"不是JSON","{}","null","42","true","[]",'"检索词"',
    '{"dense_query":null}','{"dense_query":42}','{"dense_query":true}',
    '{"dense_query":[]}','{"dense_query":{}}','{"dense_query":""}',
    '{"dense_query":" \\t\\n"}',
])
def test_rewriter_invalid_output_uses_original(monkeypatch,response):
    #1.只替换模型输出，真实解析器负责类型校验和回退
    monkeypatch.setattr("src.rag.query_rewriter.chat_completion",lambda *args,**kw: SimpleNamespace(content=response))
    result = Query_rewriter().rewrite("  查一下 splitter.py  ")
    assert result.raw_query == result.dense_query == "查一下 splitter.py"


@pytest.mark.parametrize("response",[
    '{"dense_query":"  splitter.py CodeSplitter  "}',
    '```json\n{"dense_query":"  splitter.py CodeSplitter  "}\n```',
])
def test_rewriter_valid_output_is_trimmed(monkeypatch,response):
    monkeypatch.setattr("src.rag.query_rewriter.chat_completion",lambda *args,**kw: SimpleNamespace(content=response))
    result = Query_rewriter().rewrite("怎么切代码")
    assert result.raw_query == "怎么切代码"
    assert result.dense_query == "splitter.py CodeSplitter"


@pytest.mark.parametrize("sources,expected",[
    (["rag/vectorstore.py"],{"hit":1,"rr":1.0}),
    (["other.py","rag\\vectorstore.py"],{"hit":1,"rr":0.5}),
    (["other.py"],{"hit":0,"rr":0.0}),
    ([],{"hit":0,"rr":0.0}),
])
def test_reranker_entry_enables_rerank(monkeypatch,sources,expected):
    #1.替换向量库和题集，不加载历史索引，也不调用嵌入与重排接口
    calls = []
    docs = [Document(page_content="def search(): pass",metadata={"source":source}) for source in sources]

    class FakeStore:
        def __init__(self,use_reranker=False):
            calls.append(("init",use_reranker))

        def load(self):
            calls.append(("load",))

        def search_dense(self,query,top_k):
            calls.append(("dense",query,top_k))
            return docs

        def search_bm25(self,query,top_k):
            calls.append(("bm25",query,top_k))
            return docs

        def search_hybrid(self,query,top_k):
            calls.append(("hybrid",query,top_k))
            return docs

    monkeypatch.setattr("src.rag.vectorstore.VectorStore",FakeStore)
    monkeypatch.setattr("src.evaluation.rag_dataset.EVAL_QUERIES",[
        {"query":"查找search","expected_files":["vectorstore.py"],"query_type":"exact_keyword"},
    ])
    monkeypatch.setattr("time.sleep",lambda seconds: None)
    result = runpy.run_module("src.evaluation.eval_retrieval_reranker",run_name="__main__")
    assert calls == [("init",True),("load",),("dense","查找search",8),
                     ("bm25","查找search",8),("hybrid","查找search",8)]
    assert all(row == expected for row in result["global_stats"].values())
    assert all(row == {**expected,"count":1} for row in result["stats"]["exact_keyword"].values())


def test_retry_failed_load(monkeypatch):
    instances = []

    class FakeStore:
        def __init__(self):
            instances.append(self)

        def load(self):
            if len(instances) == 1:
                raise OSError("加载失败")
            self.loaded = True

    monkeypatch.setattr(search_tool,"VectorStore",FakeStore)
    monkeypatch.setattr(search_tool,"_vector_store",None)
    with pytest.raises(OSError,match="加载失败"):
        search_tool._get_vector_store()
    assert search_tool._vector_store is None
    store = search_tool._get_vector_store()
    assert store.loaded and len(instances) == 2
    assert search_tool._get_vector_store() is store


def test_wait_for_shared_load(monkeypatch):
    started,release,second_started = Event(),Event(),Event()
    instances = []

    class FakeStore:
        def __init__(self):
            self.loaded = False
            instances.append(self)

        def load(self):
            started.set()
            assert release.wait(5)
            self.loaded = True

    def second_call():
        second_started.set()
        store = search_tool._get_vector_store()
        return store,store.loaded

    monkeypatch.setattr(search_tool,"VectorStore",FakeStore)
    monkeypatch.setattr(search_tool,"_vector_store",None)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(search_tool._get_vector_store)
        try:
            assert started.wait(5)
            assert search_tool._vector_store is None
            second = pool.submit(second_call)
            assert second_started.wait(5)
            with pytest.raises(TimeoutError):
                second.result(timeout=0.05)
        finally:
            release.set()
        store = first.result(timeout=5)
        assert second.result(timeout=5) == (store,True)
        assert len(instances) == 1


@pytest.fixture
def rerank_store(monkeypatch):
    #1.只替换嵌入和粗排来源，三条检索链路及重排逻辑仍走真实实现
    monkeypatch.setattr(settings,"embedding_provider","qwen")
    monkeypatch.setattr(settings,"reranking_provider","qwen")
    monkeypatch.setattr(settings,"qwen_api_key","test-key")
    monkeypatch.setattr(vectorstore.dashscope,"api_key","test-key")
    monkeypatch.setattr(vectorstore,"OpenAIEmbeddings",lambda **kw: object())
    store = VectorStore(use_reranker=True)
    docs = [Document(page_content="无关代码",metadata={"source":"noise.py"}),
            Document(page_content="def target(): pass",metadata={"source":"target.py"})]
    store._db = SimpleNamespace(
        similarity_search=lambda query,k: docs[:k],
        similarity_search_with_score=lambda query,k: [(doc,0.1) for doc in docs[:k]],
    )
    store.bm25_retriever = SimpleNamespace(search=lambda query,top_k: docs[:top_k])
    monkeypatch.setattr(store,"load",lambda: None)
    return store,docs


@pytest.mark.parametrize("mode",["dense","bm25","hybrid"])
@pytest.mark.parametrize("failure",["429","timeout"])
def test_rerank_failure_does_not_return_coarse_results(monkeypatch,rerank_store,mode,failure):
    #1.粗排本来有结果，重排失败必须向上传递，不能把粗排冒充重排
    store,_ = rerank_store
    def call(**kwargs):
        if failure == "timeout":
            raise TimeoutError("重排超时")
        return SimpleNamespace(status_code=429,message="请求限流")
    monkeypatch.setattr(vectorstore.dashscope.TextReRank,"call",call)
    error = TimeoutError if failure == "timeout" else RuntimeError
    with pytest.raises(error,match="超时|429"):
        getattr(store,f"search_{mode}")("target",top_k=1)


@pytest.mark.parametrize("mode",["dense","bm25","hybrid"])
def test_rerank_success_uses_returned_order(monkeypatch,rerank_store,mode):
    #1.重排把第二个候选提到第一，返回值必须真的改变顺序
    store,docs = rerank_store
    calls = []
    def call(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status_code=200,output=SimpleNamespace(results=[SimpleNamespace(index=1)]))
    monkeypatch.setattr(vectorstore.dashscope.TextReRank,"call",call)
    assert getattr(store,f"search_{mode}")("target",top_k=1) == [docs[1]]
    assert len(calls) == 1 and calls[0]["documents"] == [doc.page_content for doc in docs]


@pytest.mark.parametrize("mode",["dense","bm25","hybrid"])
def test_empty_candidates_skip_rerank_request(monkeypatch,rerank_store,mode):
    #1.没有候选不是重排失败，也不应发出空文档请求
    store,docs = rerank_store
    docs.clear()
    monkeypatch.setattr(vectorstore.dashscope.TextReRank,"call",lambda **kw: pytest.fail("空候选不应调API"))
    assert getattr(store,f"search_{mode}")("missing",top_k=1) == []


@pytest.mark.parametrize("provider,key",[("qwen",""),("unknown","test-key")])
def test_rerank_invalid_settings_fail_before_search(monkeypatch,rerank_store,provider,key):
    #1.显式要求重排时，配置不完整不能悄悄关闭重排
    monkeypatch.setattr(settings,"reranking_provider",provider)
    monkeypatch.setattr(settings,"qwen_api_key",key)
    with pytest.raises(ValueError):
        VectorStore(use_reranker=True)
    assert VectorStore(use_reranker=False).use_reranker is False


def test_local_reranker_load_failure_is_not_silenced(monkeypatch,rerank_store):
    #1.只替换可选模块，不下载本地模型；构造错误不能退回普通检索
    monkeypatch.setattr(settings,"reranking_provider","local")
    def load(name):
        raise RuntimeError("模型不可用")
    monkeypatch.setitem(sys.modules,"sentence_transformers",SimpleNamespace(CrossEncoder=load))
    with pytest.raises(RuntimeError,match="模型不可用"):
        VectorStore(use_reranker=True)


@pytest.mark.parametrize("module",["eval_retrieval","eval_retrieval_reranker"])
@pytest.mark.parametrize("failure",["429","timeout",None])
def test_retrieval_evaluation_requires_successful_rerank(tmp_path,monkeypatch,capsys,rerank_store,module,failure):
    #1.运行真实评测入口和重排函数，模拟接口结果，不访问历史索引或付费接口
    store,_ = rerank_store
    output = tmp_path / "result.json"
    calls = []
    def call(**kwargs):
        calls.append(kwargs)
        if failure == "timeout":
            raise TimeoutError("重排超时")
        if failure == "429":
            return SimpleNamespace(status_code=429,message="请求限流")
        return SimpleNamespace(status_code=200,output=SimpleNamespace(
            results=[SimpleNamespace(index=1),SimpleNamespace(index=0)]))
    def create(use_reranker=False):
        assert use_reranker is True
        return store
    monkeypatch.setattr(vectorstore,"VectorStore",create)
    monkeypatch.setattr(vectorstore.dashscope.TextReRank,"call",call)
    monkeypatch.setattr("src.evaluation.rag_dataset.EVAL_QUERIES",[
        {"query":"target","expected_files":["target.py"],"query_type":"exact_keyword"},
    ])
    monkeypatch.setattr("time.sleep",lambda seconds: None)
    for name in ("Chroma_persist_dir","bm25_persist_dir","qwen_embedding_name"):
        monkeypatch.setattr(settings,name,getattr(settings,name))
    argv = [module]
    if module == "eval_retrieval":
        argv += ["--rerank","--json-out",str(output)]
    monkeypatch.setattr(sys,"argv",argv)

    #2.失败时停止，不写结果或打印最终成绩；成功才允许标记为重排
    if failure is not None:
        error = TimeoutError if failure == "timeout" else RuntimeError
        with pytest.raises(error,match="超时|429"):
            runpy.run_module(f"src.evaluation.{module}",run_name="__main__")
        assert len(calls) == 1 and not output.exists()
        assert "最终全局评测结果" not in capsys.readouterr().out
    else:
        result = runpy.run_module(f"src.evaluation.{module}",run_name="__main__")
        assert len(calls) == 3
        if module == "eval_retrieval":
            summary = json.loads(output.read_text(encoding="utf-8"))
            assert summary["rerank"] is True
            assert all(row["mrr"] == 1.0 for row in summary["modes"].values())
        else:
            assert all(row == {"hit":1,"rr":1.0} for row in result["global_stats"].values())
