#测试bm25数据库：建库 -> 存盘 -> 重新读 -> 搜索
from langchain_core.documents import Document

from src.config import settings
from src.rag.bm25 import BM25Retriever, _tokenize


def test_bm25_index_load_search(tmp_path,monkeypatch):
    monkeypatch.setattr(settings,"bm25_persist_dir",str(tmp_path / "bm25.pkl"))
    docs = [
        Document(page_content="def index_document(self, chunks): pass",metadata={"source":"vectorstore.py"}),
        Document(page_content="def split(self, docs): pass",metadata={"source":"splitter.py"}),
        Document(page_content="class BM25Retriever: pass",metadata={"source":"bm25.py"}),
    ]
    BM25Retriever().index_document(docs)

    #新实例从硬盘读
    bm25 = BM25Retriever()
    bm25.load()
    res = bm25.search("请找到 index_document 的定义",top_k=1)
    assert res[0].metadata["source"] == "vectorstore.py"


#纯中文问题能命中相同汉字，目标放最后避免全零分排序误通过
def test_bm25_search_chinese(tmp_path,monkeypatch):
    monkeypatch.setattr(settings,"bm25_persist_dir",str(tmp_path / "bm25.pkl"))
    docs = [
        Document(page_content="log error",metadata={"source":"log.py"}),
        Document(page_content="cache value",metadata={"source":"cache.py"}),
        Document(page_content="中文检索",metadata={"source":"search.py"}),
    ]
    bm25 = BM25Retriever()
    bm25.index_document(docs)
    res = bm25.search("中文检索",top_k=1)
    assert res == [docs[2]]


#只有标点的问题分不出词，直接返回空列表
def test_bm25_search_empty_tokens(tmp_path,monkeypatch):
    monkeypatch.setattr(settings,"bm25_persist_dir",str(tmp_path / "bm25.pkl"))
    bm25 = BM25Retriever()
    bm25.index_document([Document(page_content="log error")])
    assert bm25.search("？？") == []


def test_chinese_query_has_tokens():
    assert _tokenize("如何建立向量数据库") == list("如何建立向量数据库")


def test_unrelated_query_returns_empty(tmp_path,monkeypatch):
    monkeypatch.setattr(settings,"bm25_persist_dir",str(tmp_path / "bm25.pkl"))
    bm25 = BM25Retriever()
    bm25.index_document([Document(page_content="class Store: pass"),
                         Document(page_content="def load_file(): pass")])
    assert bm25.search("火星天气预报") == []
