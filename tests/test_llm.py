#写一个小的测试文件：真的调一次大模型（要key，默认跳过：pytest -m live 才跑）
import pytest

from src.agent import llm
from src.agent.llm import chat_completion


@pytest.mark.live
@pytest.mark.parametrize("model",["deepseek-flash","deepseek-v4-pro"])
def test_chat_completion(model):
    llm.reset_usage()
    #字典列表。
    msgs = [
        {"role" : "system","content" : "你是一个研发助手，回答尽量短"},
        {"role" : "user","content" : "用一句话介绍你自己"},
    ]
    ans = chat_completion(msgs,model=model)
    print("模型回答：",ans.content)
    assert ans.content
    assert llm.USAGE["calls"] == 1 and llm.USAGE["cost_usd"] > 0
