#端到端：真实大模型 + 真实向量库（要key、要先建库，默认跳过：pytest -m live 才跑）
import json

import pytest

from src.agent.core import RepoRouter
from src.agent.tools.search_tool import search_codebase


@pytest.mark.live
def test_search_codebase():
    payload = json.loads(search_codebase("怎么建立向量数据库的类？"))
    print(payload)
    assert any("vectorstore.py" in r["source"] for r in payload["results"])


@pytest.mark.live
def test_search_private_code():
    agent = RepoRouter()
    trace = []
    res = agent.chat("请讲讲vectorstore类的定义是如何实现的",trace=trace)
    print(res)
    assert any(ev.get("tool_name") == "search_codebase" for ev in trace)


@pytest.mark.live
def test_general_question_skips_search():
    agent = RepoRouter()
    trace = []
    agent.chat("什么是快速排序？",trace=trace)
    assert not any(ev.get("tool_name") == "search_codebase" for ev in trace)
