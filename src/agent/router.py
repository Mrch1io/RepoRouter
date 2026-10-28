#Phase 2A 路由：让自己微调的小模型先“派单”
#小模型部署在OpenAI兼容接口上（本机llama.cpp，或者AutoDL上的vLLM）
import json
from typing import Optional

from openai import OpenAI
from src.config import settings
#运行时简介来自实际检索的索引，历史评测提示词快照另行保留
from lora.format_to_sharegpt import routing_system_prompt
from src.agent.tools.search_tool import _get_vector_store

ALLOWED_DECISIONS = {"tool_call","no_tool","clarify_needed"}

#_全局变量：单例，只连接一次
_client : Optional[OpenAI] = None


#懒加载：用到phase2a的时候才连接
def _get_client()->OpenAI:
    global _client
    if _client is None:
        #本地服务一般不校验key，随便填一个
        _client = OpenAI(api_key="EMPTY",base_url=settings.router_base_url,timeout=60)
    return _client


#从模型输出里找出第一个合法的路由JSON（思路和lora/eval_routing_lora.py一致）
def parse_routing_json(text : str)->dict | None:
    if not text:
        return None
    decoder = json.JSONDecoder()
    #1.从每一个 { 开始尝试解析
    i = text.find("{")
    while i != -1:
        try:
            obj,_ = decoder.raw_decode(text,i)
            #2.必须是字典，而且decision_type是三种之一
            if (
                isinstance(obj, dict)
                and isinstance(obj.get("decision_type"), str)
                and obj["decision_type"] in ALLOWED_DECISIONS
            ):
                return obj
        except json.JSONDecodeError:
            pass
        #3.不行就找下一个 {
        i = text.find("{",i + 1)
    return None


def route(question : str)->dict:
    #1.索引对象缓存简介，换库后不会沿用上一个仓库的内容
    system_prompt = routing_system_prompt(_get_vector_store().repository_summary())
    resp = _get_client().chat.completions.create(
        model=settings.router_model,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":question},
        ],
        temperature=0.1,
        max_tokens=512,
        #和训练时的 qwen3_5_nothink 模板一致：不要先“思考”，直接输出JSON（vLLM、llama.cpp 都认这个参数）
        extra_body={"chat_template_kwargs":{"enable_thinking":False}},
    )
    decision = parse_routing_json(resp.choices[0].message.content or "")
    #解析失败就当作不调工具，交给大模型直接回答（兜底，并且在trace里标出来）
    return decision or {"decision_type":"no_tool","parse_failed":True}
