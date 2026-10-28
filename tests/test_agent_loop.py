#测试Agent主循环：用假的大模型，不花钱、不联网
from types import SimpleNamespace

import pytest

import src.agent.core as core
import src.agent.router as router
from src.agent.tools.schemas import TOOLS_SCHEMA


#构造一条“模型要调工具”的回复（和SDK model_dump之后的字典一样）
def tool_reply(name,args,call_id="c1"):
    return {"role":"assistant","content":None,
            "tool_calls":[{"id":call_id,"type":"function","function":{"name":name,"arguments":args}}]}


def final_reply(text):
    return {"role":"assistant","content":text}


@pytest.fixture
def agent(monkeypatch):
    #不加载真实向量库
    monkeypatch.setattr("src.agent.tools.search_tool._get_vector_store",lambda : None)
    a = core.RepoRouter()
    #工具换成假的：记录调用参数
    a.calls = []
    for name in a.available_tools:
        a.available_tools[name] = lambda _n=name,**kw: a.calls.append((_n,kw)) or f"{_n}结果"
    return a


#让假模型按顺序返回给定的回复
def fake_llm(monkeypatch,replies):
    replies = list(replies)
    monkeypatch.setattr(core,"chat_completion",lambda *a,**kw: replies.pop(0))


def test_tool_names_match_schema(agent):
    #之前 generate_tests / generate_dockerfile 的键名和schema对不上，永远“未找到工具”
    assert set(agent.available_tools) == {t["function"]["name"] for t in TOOLS_SCHEMA}


#未知工具名在执行前拦截，并记录错误
@pytest.mark.parametrize("name",["delete_repo",[]])
def test_reject_unknown_tool(agent,name):
    trace = []
    assert "未找到工具" in agent._run_tool(name,{},trace)
    assert agent.calls == []
    assert len(trace) == 1
    assert trace[0]["error"].startswith("UnknownTool")


#参数类型、范围和字段不合法时都不执行工具
@pytest.mark.parametrize("args",[
    {"query":"VectorStore","top_k":0},
    {"query":"VectorStore","top_k":51},
    {"query":"VectorStore","top_k":"6"},
    {"top_k":6},
    {"query":"VectorStore","path":"src"},
])
def test_reject_invalid_tool_args(agent,args):
    trace = []
    assert "参数不符合工具定义" in agent._run_tool("search_codebase",args,trace)
    assert agent.calls == []
    assert len(trace) == 1
    assert trace[0]["error"].startswith("InvalidArguments")


#合法参数只执行一次，并正常记录结果
def test_run_valid_tool_call(agent):
    args = {"query":"VectorStore","top_k":6}
    trace = []
    assert agent._run_tool("search_codebase",args,trace) == "search_codebase结果"
    assert agent.calls == [("search_codebase",args)]
    assert len(trace) == 1
    assert trace[0]["tool_name"] == "search_codebase"
    assert trace[0]["arguments"] == args
    assert "error" not in trace[0]


def test_no_tool_direct_answer(agent,monkeypatch):
    fake_llm(monkeypatch,[final_reply("快速排序是……")])
    trace = []
    assert agent.chat("什么是快速排序",trace=trace) == "快速排序是……"
    assert agent.calls == []
    assert trace[-1]["event"] == "usage"   #每道题都记录花费和耗时


def test_tool_call_then_answer(agent,monkeypatch):
    fake_llm(monkeypatch,[tool_reply("analyse_code",'{"code": "x=1"}'),final_reply("没问题")])
    assert agent.chat("分析这段代码 x=1") == "没问题"
    assert agent.calls == [("analyse_code",{"code":"x=1"})]
    #记忆修剪：工具调用的中间消息不留在记忆里
    assert all(m["role"] != "tool" for m in agent.memory)


def test_block_only_duplicate_calls(agent,monkeypatch):
    fake_llm(monkeypatch,[
        tool_reply("search_codebase",'{"query": "VectorStore"}',"c1"),
        tool_reply("search_codebase",'{"query": "VectorStore"}',"c2"),   #完全一样：拦截
        tool_reply("search_codebase",'{"query": "BM25"}',"c3"),          #换了关键词：放行
        final_reply("好了"),
    ])
    agent.chat("向量库在哪")
    assert [kw["query"] for _,kw in agent.calls] == ["VectorStore","BM25"]


def test_stop_after_five_rounds(agent,monkeypatch):
    #1.窗口只留一轮，停止提示也要与本轮问题一起保留，不能留下旧回答
    monkeypatch.setattr(core.settings,"routing_backend","native")
    system = {"role":"system","content":"原来的系统提示"}
    agent.memory = [system,{"role":"user","content":"旧问题"},final_reply("旧回答")]
    fake_llm(monkeypatch,[tool_reply("search_codebase",f'{{"query": "q{i}"}}',f"c{i}") for i in range(5)])
    stopped = agent.chat("一直查")
    assert "最大迭代次数" in stopped
    history = [system,{"role":"user","content":"一直查"},final_reply(stopped)]
    assert agent.memory == history
    assert len(agent.calls) == 5

    #2.下一问只带完整的上一轮，回答后再按轮裁剪
    seen = []
    monkeypatch.setattr(core,"chat_completion",lambda msgs,**kw: seen.extend(msgs) or final_reply("新的回答"))
    assert agent.chat("换个问题") == "新的回答"
    assert seen == history + [{"role":"user","content":"换个问题"}]
    assert agent.memory == [system,{"role":"user","content":"换个问题"},final_reply("新的回答")]


#第一次请求和已有多轮历史时，连续失败都不能留下问题；恢复后只看到成功历史
@pytest.mark.parametrize("backend",["native","phase2a"])
@pytest.mark.parametrize("has_history",[False,True])
def test_failed_requests_preserve_memory(agent,monkeypatch,backend,has_history):
    monkeypatch.setattr(core.settings,"routing_backend",backend)
    monkeypatch.setattr(router,"route",lambda q: {"decision_type":"no_tool"})
    agent.max_history_turns = 4
    if has_history:
        agent.memory = [{"role":"system","content":"原来的系统提示"}]
        for i in range(4):
            agent.memory += [{"role":"user","content":f"旧问题{i}"},
                             {"role":"assistant","content":f"旧回答{i}"}]
    history = list(agent.memory)

    def timeout(*args,**kwargs):
        raise TimeoutError("模拟超时")

    monkeypatch.setattr(core,"chat_completion",timeout)
    for i in range(12):
        with pytest.raises(TimeoutError,match="模拟超时"):
            agent.chat(f"失败问题{i}")
        assert agent.memory == history

    seen = []
    reply = final_reply("恢复了") if backend == "native" else SimpleNamespace(content="恢复了")
    monkeypatch.setattr(core,"chat_completion",lambda messages,**kw: seen.extend(messages) or reply)
    assert agent.chat("恢复请求") == "恢复了"
    assert [m["content"] for m in seen if m["role"] == "user"] == [m["content"] for m in history if m["role"] == "user"] + ["恢复请求"]
    assert len(agent.memory) <= 2 * agent.max_history_turns + 1


#工具已经执行、最终回答失败时，问题和中间工具消息也一起回滚
@pytest.mark.parametrize("backend",["native","phase2a"])
def test_failed_tool_answer_preserves_memory(agent,monkeypatch,backend):
    monkeypatch.setattr(core.settings,"routing_backend",backend)
    monkeypatch.setattr(router,"route",lambda q: {"decision_type":"tool_call","tool_name":"analyse_code","arguments":{"code":"x=1"}})
    history = [{"role":"system","content":"原来的系统提示"},
               {"role":"user","content":"旧问题"},{"role":"assistant","content":"旧回答"}]
    agent.memory = list(history)

    def answer(*args,**kwargs):
        if not agent.calls:
            return tool_reply("analyse_code",'{"code":"x=1"}')
        raise TimeoutError("回答超时")

    monkeypatch.setattr(core,"chat_completion",answer)
    with pytest.raises(TimeoutError,match="回答超时"):
        agent.chat("分析 x=1")
    assert agent.calls == [("analyse_code",{"code":"x=1"})]
    assert agent.memory == history


#小模型路由本身失败时，也不能把未完成的问题留给下次请求
def test_failed_route_preserves_memory(agent,monkeypatch):
    monkeypatch.setattr(core.settings,"routing_backend","phase2a")

    def timeout(question):
        raise TimeoutError("路由超时")

    monkeypatch.setattr(router,"route",timeout)
    with pytest.raises(TimeoutError,match="路由超时"):
        agent.chat("这次失败")
    assert agent.memory == []


def test_router_runs_tool(agent,monkeypatch):
    monkeypatch.setattr(core.settings,"routing_backend","phase2a")
    monkeypatch.setattr(router,"route",lambda q: {"decision_type":"tool_call","tool_name":"analyse_log","arguments":{"log":"Error"}})
    seen = []
    monkeypatch.setattr(core,"chat_completion",lambda msgs,**kw: seen.append(msgs) or SimpleNamespace(content="是空指针"))
    assert agent.chat("帮我看看这个报错 Error") == "是空指针"
    assert agent.calls == [("analyse_log",{"log":"Error"})]
    assert "analyse_log结果" in seen[0][-1]["content"]   #工具结果交给了大模型


#回答阶段只换消息副本的提示，不改记忆里的共享提示
def test_router_no_tool_prompt(agent,monkeypatch):
    monkeypatch.setattr(core.settings,"routing_backend","phase2a")
    monkeypatch.setattr(router,"route",lambda q: {"decision_type":"no_tool","tool_name":None})
    seen = []
    monkeypatch.setattr(core,"chat_completion",lambda msgs,**kw: seen.append((msgs,kw)) or SimpleNamespace(content="快速排序是……"))

    assert agent.chat("什么是快速排序") == "快速排序是……"
    assert len(seen) == 1
    messages,kwargs = seen[0]
    assert messages[0]["role"] == "system"
    assert "是否使用工具已经由路由决定" in messages[0]["content"]
    assert "必须毫不犹豫地调用对应工具" not in messages[0]["content"]
    assert "必须毫不犹豫地调用对应工具" in agent.memory[0]["content"]
    assert messages[0] is not agent.memory[0]
    assert "tools" not in kwargs
    assert agent.calls == []


#回答里的工具调用文字只记入trace，答案仍然原样返回
@pytest.mark.parametrize("answer,flagged",[
    ('<tool_call>{"name":"search_codebase"}</tool_call>',True),
    ('我先调用 search_codebase("VectorStore")',True),
    ('我先调用 analyse_log("Error")',True),
    ("当前没有检索到仓库信息。",False),
])
def test_router_logs_tool_call_text(agent,monkeypatch,answer,flagged):
    monkeypatch.setattr(core.settings,"routing_backend","phase2a")
    monkeypatch.setattr(router,"route",lambda q: {"decision_type":"no_tool","tool_name":None})
    monkeypatch.setattr(core,"chat_completion",lambda msgs,**kw: SimpleNamespace(content=answer))
    trace = []

    assert agent.chat("VectorStore在哪里",trace=trace) == answer
    events = [event for event in trace if event.get("event") == "answer_tool_call_text"]
    assert events == ([{"event":"answer_tool_call_text"}] if flagged else [])
    assert agent.calls == []


@pytest.mark.parametrize("answer",["请贴出完整报错"," 请贴出完整报错\n"])
def test_router_clarifies(agent,monkeypatch,answer):
    monkeypatch.setattr(core.settings,"routing_backend","phase2a")
    monkeypatch.setattr(router,"route",lambda q: {"decision_type":"clarify_needed","clarify_answer":answer})
    fake_llm(monkeypatch,[])
    assert agent.chat("报错了") == answer
    assert agent.memory[-1] == {"role":"assistant","content":answer}
    assert agent.calls == []


@pytest.mark.parametrize("answer",[None,{"text":"请补充"},["请补充"],1,True,""," \n\t"])
def test_router_clarify_fallback(agent,monkeypatch,answer):
    monkeypatch.setattr(core.settings,"routing_backend","phase2a")
    monkeypatch.setattr(router,"route",lambda q: {"decision_type":"clarify_needed","clarify_answer":answer})
    fake_llm(monkeypatch,[])
    assert agent.chat("报错了") == "请补充更多信息。"
    assert agent.memory[-1] == {"role":"assistant","content":"请补充更多信息。"}
    assert agent.calls == []


def test_router_parse():
    assert router.parse_routing_json('好的 {"decision_type":"no_tool","tool_name":null}')["decision_type"] == "no_tool"
    assert router.parse_routing_json('{"decision_type":"乱写"}') is None
    assert router.parse_routing_json("") is None


def test_check_tool_args():
    #BFCL 式参数检查：不要求和金标逐字一样，只看是否合法
    from src.agent.tools.schemas import check_tool_args
    assert check_tool_args("search_codebase",{"query":"VectorStore","top_k":6})
    assert check_tool_args("generate_tests",{"code":"def f(): pass"})          #可选参数可以不给
    assert not check_tool_args("search_codebase",{"top_k":6})                 #缺必填参数
    assert not check_tool_args("search_codebase",{"query":"  "})              #必填字符串是空的
    assert not check_tool_args("search_codebase",{"query":"x","top_k":"6"})   #类型不对
    assert not check_tool_args("search_codebase",{"query":"x","top_k":True})  #bool 不能当 int
    assert not check_tool_args("search_codebase",{"query":"x","path":"src"})  #编出来的参数
    assert not check_tool_args("delete_repo",{"x":1})                          #工具不存在


def test_knn_router_vote():
    #向量近邻基线：最像的例题投票；留一法时不能投给自己
    from lora.eval_routing_knn import knn_predict,_normalize
    golds = [{"decision_type":"tool_call","tool_name":"search_codebase"},
             {"decision_type":"no_tool","tool_name":None},
             {"decision_type":"no_tool","tool_name":None}]
    vecs = [_normalize(v) for v in ([1,0],[0,1],[0.1,1])]
    assert knn_predict([_normalize([1,0.05])],vecs,golds,k=1)[0]["decision_type"] == "tool_call"
    assert knn_predict(vecs,vecs,golds,k=1,skip_self=True)[0]["decision_type"] == "no_tool"


def test_usage_cost():
    from src.agent import llm
    llm.reset_usage()
    usage = SimpleNamespace(prompt_tokens=1000,completion_tokens=500,prompt_cache_hit_tokens=800,prompt_cache_miss_tokens=200)
    llm._add_usage(usage,"deepseek-flash")
    assert llm.USAGE["calls"] == 1
    assert llm.USAGE["cost_usd"] == pytest.approx((800*0.006 + 200*0.3 + 500*1.2) / 1e6)
