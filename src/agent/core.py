import time




from src.agent.llm import chat_completion, USAGE, reset_usage
from src.config import settings

#导入工具
import json
import re
from src.agent.tools.schemas import TOOLS_SCHEMA, check_tool_args
from src.agent.tools.code_analyser import analyse_code
from src.agent.tools.search_tool import search_codebase, time_logger
from src.agent.tools.code_completion import code_completion
from src.agent.tools.dockerfile import gen_docker_file
from src.agent.tools.test_generation import gen_tests
from src.agent.tools.log_analysis import analyse_log

def safe_json_loads(s: str) -> dict:
    
    if not isinstance(s, str): 
        return {}
    s = s.strip()
    # 去除可能存在的 ```json 和 ``` 标记
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # 裁剪出第一个 { 到最后一个 } 的内容
    left, right = s.find("{"), s.rfind("}")
    if left != -1 and right != -1 and right > left:
        s = s[left:right+1]
    try:
        return json.loads(s)
    except Exception as e:
        print(f" [JSON 解析失败] 原始字符串: {s} \n错误: {e}")
        return {}
#定义成类而非函数

#创建实例，维护状态变量

#而且函数需要每次调用重新初始化，类一次初始化构建。
class RepoRouter:
    #核心RepoRouter


    
                #加入多轮记忆
    def __init__(self,max_history_turns = 1):
        #构建工具映射字典：
        self.available_tools = {
            "analyse_code":analyse_code,
            "complete_code":code_completion,
            #键名必须和 schemas.py 里的 name 一模一样，否则永远“未找到工具”
            "generate_dockerfile":gen_docker_file,
            "search_codebase":search_codebase,
            "generate_tests":gen_tests,
            "analyse_log":analyse_log,
        }
        #初始化对话记忆数组
        self.memory = []
        self.max_history_turns = max_history_turns
    
        #提前触发单例加载
        from src.agent.tools.search_tool import _get_vector_store
        _get_vector_store()

    #多轮记忆去粗取精滑动窗口
    def _prune_memory(self):
        if not self.memory:
            return
        
        pruned_chat_history = []
        #1.留下system prompt
        #传入的是字典格式
        system_prompt = self.memory[0] if self.memory[0].get("role") == "system" else None

        #2.过滤中间的tool calls and observations
        for msg in self.memory:
            role = msg.get("role")

            if role == "tool" or(role == "assistant" and msg.get("tool_calls")):
                continue #这条不要
            elif role == "system":
                continue
            pruned_chat_history.append(msg)

        #3.去掉旧记忆
        max_len = 2*self.max_history_turns
        if len(pruned_chat_history) > max_len:
            print(f"目前已达设定上下文上限{self.max_history_turns}轮，正在裁剪记忆\n")
            pruned_chat_history = pruned_chat_history[-max_len:]
            


        #4.重新组装
        new_memory = []
        if system_prompt:
            new_memory.append(system_prompt)
        new_memory.extend(pruned_chat_history) #字典列表：先解包再放进去
        
        self.memory = new_memory
        print(f"[Memory] 修剪完毕，过滤掉工具调用残余，当前保留 {len(self.memory)} 条核心消息。")

    #每道题的token、花费、耗时，写进trace给网页和评测用
    def _log_usage(self,trace,start):
        if trace is not None:
            trace.append({"event":"usage",**USAGE,"seconds":round(time.perf_counter() - start,2)})

    #执行一个工具，并把结果记进trace（native和phase2a两条路共用）
    def _run_tool(self,function_name,args,trace)->str:
        tool_error = None

        if not isinstance(function_name, str) or function_name not in self.available_tools:
            tool_error = f"UnknownTool: {function_name}"
            function_resp = f"未找到工具{function_name}\n"
            print(function_resp)
        elif not check_tool_args(function_name, args):
            tool_error = f"InvalidArguments: {function_name}"
            function_resp = f"工具'{function_name}'参数不符合工具定义，未执行。请检查必填参数、类型和取值范围。"
            print(function_resp)
        else:
            function_to_call = self.available_tools[function_name]
            try:
                #执行代码
                function_resp = function_to_call(**args) #解包
            except Exception as e:
                tool_error = f"{type(e).__name__}: {e}"
                #给function_resp一个可用值
                function_resp = f"工具'{function_name}'内部执行出错\n错误信息：{str(e)}"
                print(function_resp)

        #2.记录trace
        if trace is not None:
            ev: dict = {
                "tool_name": function_name,
                "arguments": args,
                "result_preview": str(function_resp)[:2000],
            }
            if tool_error:
                ev["error"] = tool_error
            if function_name == "search_codebase":
                try:
                    ev["search_payload"] = json.loads(str(function_resp))
                except Exception:
                    ev["search_payload"] = None
            trace.append(ev)
        return str(function_resp) #保证转回字符串

    #phase2a：小模型先派单（要不要调工具、调哪个），大模型只负责写最终回答
    def _chat_phase2a(self,trace,max_tokens=800)->str:
        from src.agent.router import route
        #1.小模型派单
        question = self.memory[-1]["content"]
        history = [msg for msg in self.memory[:-1] if msg["role"] in {"user","assistant"} and not msg.get("tool_calls")]
        if history:
            #历史只供理解指代，仍让单轮路由模型只给当前问题派单
            question = ("【历史对话（只用于理解本轮指代）】\n"
                        f"{json.dumps(history,ensure_ascii=False)}\n"
                        f"【本轮问题】\n{question}\n请只对本轮问题输出路由JSON。")
        decision = route(question)
        if trace is not None:
            trace.append({"event":"router_decision","decision":decision})

        dt = decision.get("decision_type")
        if dt == "clarify_needed":
            #2.信息不够：直接返回追问
            answer = decision.get("clarify_answer")
            if not isinstance(answer,str) or not answer.strip():
                answer = "请补充更多信息。"
        else:
            #复制一份，工具结果只在本次请求里用，不进记忆
            messages = list(self.memory)
            #2.只替换副本里的系统提示，保留记忆中的原提示
            messages[0] = {"role":"system","content":(
                "是否使用工具已经由路由决定，你只负责生成最终回答。\n"
                "有工具结果时，依据工具结果回答。\n"
                "没有工具结果时，基于通用知识回答；如果问题涉及仓库具体实现而缺少依据，就说明当前没有检索到仓库信息，不要编造。\n"
                "不要输出工具调用，也不要声称要去调用工具。\n"
                "用中文回答。"
            )}
            if dt == "tool_call":
                #3.按派单执行工具，把结果交给大模型
                name = decision.get("tool_name")
                result = self._run_tool(name,decision.get("arguments"),trace)
                messages.append({"role":"user","content":f"工具 {name} 的返回结果如下，请据此回答上面的问题：\n{result}"})
                if trace is not None:
                    trace.append({"event": "tool_context", "tool_name": name, "content": result})
            #4.不给tools参数：大模型不再自己决定调工具，只写回答
            answer = chat_completion(messages, max_tokens=max_tokens, trace=trace).content or ""

            #5.记录回答里的工具调用字样，答案照常返回
            if trace is not None and ("<tool_call>" in answer or any(f"{name}(" in answer for name in self.available_tools)):
                trace.append({"event":"answer_tool_call_text"})

        self.memory.append({"role":"assistant","content":answer})
        self._prune_memory()
        return answer

    @time_logger              #类型声明清楚:str
    def chat(self,question : str,trace : list | None = None,max_tokens : int = 800)->str:
        #1.只复制列表，已有消息内容不变，不用深拷贝
        history = list(self.memory)
        try:
            return self._chat(question,trace,max_tokens)
        except Exception:
            #2.失败退回本轮开始前，异常照常抛给调用方
            self.memory = history
            raise

    def _chat(self,question : str,trace : list | None,max_tokens : int)->str:
        reset_usage()
        start = time.perf_counter()

        #如果是第一轮对话，放入system prompt
        if not self.memory:
            print("加载系统提示词")
            system_prompt = """你是一个拥有多智能体协作能力的研发助手。
            你有多个的辅助工具（如静态代码分析、源码检索、日志分析等）。
            请严格遵守以下指令：
            1. 先判断用户问题是否需要依赖“当前私有代码仓库”的具体实现信息。
            2. 分析用户问题，如果需要查阅代码、分析 Bug 或生成配置，必须毫不犹豫地调用对应工具。
            3. 如果问题属于通用编程知识、算法基础、框架基础概念、常见工程经验（例如“什么是快速排序”“FastAPI 路由怎么注册”“什么是 BM25”），禁止调用 search_codebase，直接基于你的通用知识回答。
            4. 只有当问题明确涉及当前代码仓库中的文件、函数、类、内部实现逻辑、配置方式、私有模块关系，或缺少私有上下文就无法可靠回答时，才允许调用 search_codebase。
            5. 除非明显缺少信息，否则不要重复调用工具。每个问题最多调用 1 次 search_codebase。
            6. 必须用中文回答。如果工具返回“未命中高相关度私有知识”，则不要继续调用 search_codebase，应直接基于通用知识回答，或明确说明当前仓库中未检索到相关实现。
            """
            self.memory.append({"role":"system","content":system_prompt})
        #放入用户提问
        self.memory.append({"role":"user","content":question})

        #路由换成自己微调的小模型
        if settings.routing_backend == "phase2a":
            answer = self._chat_phase2a(trace, max_tokens=max_tokens)
            self._log_usage(trace, start)
            return answer

        #引入调用指纹：防止相同参数循环
        call_signatures = set()

        for _ in range(5):
            print("Agent综合分析中\n")

            #1.呼出大模型
            response_obj = chat_completion(self.memory,tools=TOOLS_SCHEMA, max_tokens=max_tokens, trace=trace)
            #返回的是SDK对象,转换为纯字典

            try:
                resp_dict = response_obj.model_dump(exclude_unset = True) #去掉None内容
            except AttributeError:
                #如果前面已经转为了字典
                resp_dict = response_obj
            #把这一步关于工具调用的分析记录下来

            #已经加入记忆
            self.memory.append(resp_dict)

            #2.判断是否需要调用工具

            tool_calls = resp_dict.get("tool_calls",None)

            if tool_calls:
                if trace is not None:
                    trace.append(
                        {
                            "event": "assistant_tool_calls",
                            "names": [tc["function"]["name"] for tc in tool_calls],
                        }
                    )
                #遍历所有要的工具
                for tool_call in tool_calls:
                    #获取名字
                    function_name = tool_call["function"]["name"]
                    #获取参数
                    argument_str = tool_call["function"]["arguments"]
                    args = safe_json_loads(argument_str)
                    #指纹 = 工具名 + 参数（排好序再转字符串）。之前只用工具名，换了关键词也会被误拦
                    call_fingerprint = f"{function_name}_{json.dumps(args, sort_keys=True, ensure_ascii=False)}"

                    if call_fingerprint in call_signatures:
                        #发现大模型在原地踏步（用完全一样的参数调同一个工具）！
                        print(f"检测到重复调用: {function_name}")
                        
                        # 
                        error_msg = "系统拦截：你刚刚已经调用过该工具了！请不要原地踏步。如果你已经收集到足够信息，请立刻输出最终回答；如果信息不够，请更换检索关键词！"
                        
                        self.memory.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": error_msg
                        })
                        if trace is not None:
                            trace.append({"event": "tool_context", "tool_name": function_name, "content": error_msg})
                        continue # 让它带着指令，进入下一次循环重新思考！
                    #如果不在，加入集合
                    
                    call_signatures.add(call_fingerprint)

                    print(f"Agent决策：决定调用工具：{function_name}")
                    function_resp = self._run_tool(function_name, args, trace)

                    #调用完工具，将observation放回去
                    tool_msg = {
                        "role":"tool",
                        "tool_call_id" : tool_call["id"],
                        "content":function_resp,
                    }
                    self.memory.append(tool_msg)
                    if trace is not None:
                        trace.append({"event": "tool_context", "tool_name": function_name, "content": function_resp})
                #一轮跑完继续循环，让模型可以对着结果再思考
                continue
            else:
                #没有tool_calls，模型觉得不用调用工具，
                #直接提取回复的人话
                final_resp = resp_dict.get("content") or ""

                #记忆修剪
                self._prune_memory()
                if trace is not None:
                    trace.append({"event": "final", "tool_calls_in_turn": False})
                self._log_usage(trace, start)
                return final_resp

        #超过次数也要修剪，否则下一轮会带着没配对的工具消息
        final_resp = "Agent思考超过最大迭代次数（5次）已停止思考....."
        self.memory.append({"role":"assistant","content":final_resp})
        self._prune_memory()
        if trace is not None:
            trace.append({"event": "answer_metadata", "finish_reason": "iteration_limit"})
        self._log_usage(trace, start)
        return final_resp

    def clear_memory(self) -> None:
        self.memory = []
