#测试FastAPI接口：不联网，大模型相关的函数都换成假的
import pytest
from fastapi.testclient import TestClient

import src.api.routes as routes
from src.main import app

client = TestClient(app)


class FakeAgent:
    def __init__(self,max_history_turns=1):
        self.history = []
    def chat(self,question):
        self.history.append(question)
        return f"第{len(self.history)}问：{question}"
    def clear_memory(self):
        self.history = []


def test_health():
    assert client.get("/health").json() == {"status":"OK"}


def test_session_isolation(monkeypatch):
    monkeypatch.setattr(routes,"RepoRouter",FakeAgent)
    monkeypatch.setattr(routes,"_agents",{})
    client.post("/api/v1/chat",json={"question":"a","session_id":"u1"})
    r = client.post("/api/v1/chat",json={"question":"b","session_id":"u2"})
    #不同用户各自一份记忆，不会串话
    assert r.json()["answer"] == "第1问：b"
    assert client.post("/api/v1/chat",json={"question":"c","session_id":"u1"}).json()["answer"] == "第2问：c"
    assert routes._agents["u1"].history == ["a","c"]
    assert routes._agents["u2"].history == ["b"]


@pytest.mark.parametrize("session",[{}, {"session_id":None}, {"session_id":""},
                                     {"session_id":" \t\n"}, {"session_id":"\u3000"}])
def test_chat_requires_session_id(monkeypatch,session):
    #1.省略、空串和纯空白都不进入Agent，也就不会共享默认会话
    monkeypatch.setattr(routes,"RepoRouter",FakeAgent)
    monkeypatch.setattr(routes,"_agents",{})
    r = client.post("/api/v1/chat",json={"question":"a",**session})
    assert r.status_code == 422
    assert routes._agents == {}


@pytest.mark.parametrize("params",[{}, {"session_id":""}, {"session_id":" \t\n"},
                                   {"session_id":"\u3000"}])
def test_clear_requires_session_id(monkeypatch,params):
    #1.清空接口也必须明确指定会话
    monkeypatch.setattr(routes,"RepoRouter",FakeAgent)
    monkeypatch.setattr(routes,"_agents",{})
    assert client.post("/api/v1/clear-memory",params=params).status_code == 422
    assert routes._agents == {}


def test_clear_unknown_session_preserves_full_cache(monkeypatch):
    #1.填满会话，清空一个不存在的ID不应创建或驱逐任何对象
    agents = {f"u{i}":FakeAgent() for i in range(routes.MAX_SESSIONS)}
    for session_id,agent in agents.items():
        agent.chat(session_id)
    original = dict(agents)
    monkeypatch.setattr(routes,"_agents",agents)
    monkeypatch.setattr(routes,"RepoRouter",lambda **kw: pytest.fail("清空不应创建Agent"))
    r = client.post("/api/v1/clear-memory",params={"session_id":"never-created"})
    assert r.status_code == 200 and r.json() == {"status":"OK"}
    assert routes._agents == original
    assert all(agent.history == [session_id] for session_id,agent in agents.items())


def test_clear_only_selected_session(monkeypatch):
    #1.已有会话只清目标，重复清空也不会影响另一份历史
    first,second = FakeAgent(),FakeAgent()
    first.chat("a")
    second.chat("b")
    monkeypatch.setattr(routes,"_agents",{"u1":first,"u2":second})
    for _ in range(2):
        assert client.post("/api/v1/clear-memory",params={"session_id":"u1"}).status_code == 200
    assert first.history == []
    assert second.history == ["b"]
    assert routes._agents == {"u1":first,"u2":second}


@pytest.mark.parametrize("code,expected",[
    ("\ndef f(x=[]):\n    return x\n",("第2行","函数'f'使用了默认可变参数 []","错误：必要修复")),
    ('try:\n    value = int("1")\nexcept:\n    value = 0\n',("第3行","使用了裸except","警告：建议修复")),
    ("def f(:\n    pass",("第1行","语法错误","错误：必要修复")),
    ("def f(x):\n    return x + 1\n",("代码检查完成 未发现问题",)),
])
def test_analyse_code_is_real(code,expected):
    #1.走真实接口和分析器，恒定返回OK或同一份错误报告都不能通过
    r = client.post("/api/v1/analyse-code",json={"code":code})
    assert r.status_code == 200
    report = r.json()["analyse"]
    assert all(text in report for text in expected)


def test_dockerfile_and_log(monkeypatch):
    monkeypatch.setattr(routes,"gen_docker_file",lambda d: "FROM python:3.12-slim")
    monkeypatch.setattr(routes,"analyse_log",lambda log: "空指针")
    assert client.post("/api/v1/generate-dockerfile",json={"project_description":"fastapi"}).json()["docker-file"].startswith("FROM")
    assert client.post("/api/v1/log-analyse",json={"log_content":"Error"}).json()["log-analyse"] == "空指针"


def test_web_details_survive_rerun(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    class WebAgent(FakeAgent):
        def chat(self,question,trace):
            answer = super().chat(question)
            trace.extend([
                {"event":"router_decision","decision":{"decision_type":"tool_call","tool_name":"search_codebase"}},
                {"tool_name":"search_codebase","arguments":{"query":question},"result_preview":"代码",
                 "search_payload":{"results":[{"source":f"file{len(self.history)}.py","lines":"1-3"}]}},
                {"event":"tool_context","tool_name":"search_codebase","content":"代码"},
                {"event":"tool_context","tool_name":"search_codebase","content":"系统拦截重复调用"},
                {"tool_name":"analyse_code","arguments":{},"result_preview":"参数无效","error":"InvalidArguments: analyse_code"},
                {"event":"usage","cost_usd":0.0015 * len(self.history)},
            ])
            return answer

    #1.渲染真实网页代码，只替换后端，不连接向量库和模型
    monkeypatch.setattr("src.agent.core.RepoRouter",WebAgent)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "web_ui.py").run()
    app.chat_input[0].set_value("第一个问题").run()
    assert not app.exception
    first_label = app.get("status")[0].label
    assert "$0.0015" in first_label
    saved = dict(app.session_state.messages[-1])
    assert saved["elapsed_time"] >= 0 and saved["cost"] == 0.0015

    #2.再问一次，旧状态框与耗时费用原样保留；回传和被拦截的调用不重复计数
    app.chat_input[0].set_value("第二个问题").run()
    app.run()
    assert not app.exception
    statuses = app.get("status")
    assert len(statuses) == 2 and statuses[0].label == first_label
    assert "$0.0030" in statuses[1].label
    assert app.session_state.messages[1] == saved
    for status in statuses:
        text = [item.value for item in status.get("markdown")]
        assert [s for s in text if s.startswith("调用工具：")] == ["调用工具：`search_codebase`"]
        assert any("工具处理失败：`analyse_code`" in s for s in text)
        assert any("第 1-3 行" in s for s in text)

    #3.清空对话时，正文和已保存的详情一起清掉
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state.messages == [] and len(app.get("status")) == 0


def test_web_accepts_old_messages(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    #1.旧会话没有trace和耗时字段，仍然只显示原正文
    monkeypatch.setattr("src.agent.core.RepoRouter",FakeAgent)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "web_ui.py").run()
    app.session_state.messages = [{"role":"assistant","content":"旧回答"}]
    app.run()
    assert not app.exception
    assert any(item.value == "旧回答" for item in app.markdown)
    assert len(app.get("status")) == 0


@pytest.mark.parametrize("has_history",[False,True])
def test_web_failed_request_can_retry(monkeypatch,has_history):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import src.agent.core as core

    #1.用真实Agent处理记忆，只替换模型和向量库，先成功一次再模拟超时
    monkeypatch.setattr("src.agent.tools.search_tool._get_vector_store",lambda: None)
    monkeypatch.setattr(core.settings,"routing_backend","native")
    replies = (["旧回答"] if has_history else []) + [TimeoutError("模拟超时"),"重试成功"]
    seen = []

    def answer(messages,**kwargs):
        seen.append(list(messages))
        reply = replies.pop(0)
        if isinstance(reply,Exception):
            raise reply
        return {"role":"assistant","content":reply}

    monkeypatch.setattr(core,"chat_completion",answer)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "web_ui.py").run()
    if has_history:
        app.chat_input[0].set_value("旧问题").run()
    messages = list(app.session_state.messages)
    memory = list(app.session_state.agent.memory)

    #2.失败会显示提示，前后端历史都不留下半轮对话
    app.chat_input[0].set_value("再试一次").run()
    assert not app.exception
    assert len(app.error) == 1 and "请稍后重新发送" in app.error[0].value
    assert app.get("status")[-1].state == "error"
    assert app.session_state.messages == messages
    assert app.session_state.agent.memory == memory

    #3.同一问题重试成功后只保存一次，刷新也不再显示失败消息
    app.chat_input[0].set_value("再试一次").run()
    app.run()
    assert not app.exception and not app.error
    assert seen[-1] == seen[-2]
    expected = [{"role":msg["role"],"content":msg["content"]} for msg in messages]
    expected += [{"role":"user","content":"再试一次"},{"role":"assistant","content":"重试成功"}]
    assert [{"role":msg["role"],"content":msg["content"]} for msg in app.session_state.messages] == expected
    assert app.session_state.agent.memory[1:] == expected
    assert [item.value for item in app.markdown].count("再试一次") == 1
