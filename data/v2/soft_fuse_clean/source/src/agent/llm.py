#我们有很多不同请求，都是要给LLM发送信息，接收信息
#把发送请求的逻辑全部封装起来
from typing import List,Dict,Optional #类型标注：List,Dict
from src.config import settings #配置文件

#工业界标准openai接口
from openai import OpenAI

#实例化官方客户端
#没填key时给个占位：import不报错（跑测试、CI用），真正调用时才会报401
client = OpenAI(
    api_key=settings.deepseek_api_key or "missing-key",
    base_url=settings.deepseek_base_url,
    timeout=120,      #网络卡住时不要一直等
    max_retries=2,
)

#DeepSeek官方价格（美元/百万token，评测时固定单价，按高峰价算，偏保守）
#顺序：[缓存命中的输入,未命中的输入,输出]
PRICES = {
    "deepseek-flash":[0.006,0.3,1.2],
    "deepseek-v4-pro":[0.044,1.32,3.96],
}

#用量记录：每道题开始时清零，结束时读出来算花费
#全局变量：单人演示够用；多人同时用会混在一起
USAGE = {"calls":0,"cache_hit":0,"cache_miss":0,"output":0,"cost_usd":0.0}

def reset_usage():
    for k in USAGE:
        USAGE[k] = 0


#每调用一次大模型，累加一次用量
def _add_usage(usage,model : str):
    if usage is None:
        return
    #1.DeepSeek会单独返回缓存命中/未命中的token数，其他接口没有就自己算
    hit = getattr(usage,"prompt_cache_hit_tokens",0) or 0
    miss = getattr(usage,"prompt_cache_miss_tokens",None)
    if miss is None:
        miss = usage.prompt_tokens - hit
    #2.按价格表算钱（不在表里的模型按0算）
    price = PRICES.get(model,[0,0,0])
    USAGE["calls"] += 1
    USAGE["cache_hit"] += hit
    USAGE["cache_miss"] += miss
    USAGE["output"] += usage.completion_tokens
    USAGE["cost_usd"] += (hit*price[0] + miss*price[1] + usage.completion_tokens*price[2]) / 1e6


#函数输入
#对话补全
def chat_completion(
        #需要传入的参数
        #上下文对话

        #大模型本身没有记忆，记忆由后端代码维护
        messages : List[Dict[str,str]],#一个字典列表：system:.....user:....：完整对话/
        #冒号就是标注类型，等价于int max_tokens = 800。一般的传参就是(messages,max_tokens = 800,temperature = 0.2)
        max_tokens: int = 800,
        temperature : float = 0.2,
        tools = None,
        model : Optional[str] = None,   #不传就用配置里的模型，对比实验时可以换
        trace: list | None = None,

):
    """
    调用deepseek api接口
    messages诸如：{"role" : "user","content" : "......"}
                    {"role" : "system","content" : "......"}
    返回 message 对象（含 content、tool_calls 等），不是字符串
    """
    model = model or settings.model_name
    #没有工具就不传tools参数
    kwargs = {}
    if tools:
        kwargs["tools"] = tools

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        #V4的思考模式开关，要放在extra_body里
        extra_body={"thinking":{"type":"enabled" if settings.llm_thinking else "disabled"}},
        **kwargs,
    )
    _add_usage(response.usage,model)
    if trace is not None:
        trace.append({"event": "answer_metadata", "finish_reason": response.choices[0].finish_reason})
    return response.choices[0].message
"""{
  "id": "chatcmpl-123",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "死锁是指两个以上的进程..."  <--- 我们真正想要的在这！
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 10, "completion_tokens": 20}
}"""
