from typing import Dict#类型标注
from fastapi import HTTPException,APIRouter,Query#分发网址
from pydantic import BaseModel,Field
from src.agent.core import RepoRouter
from src.agent.tools.code_analyser import analyse_code
from src.agent.tools.dockerfile import gen_docker_file
from src.agent.tools.log_analysis import analyse_log
#pydantic:数据校验库：字典里面的数据对不对（检查格式等等）



#在routes.py建立一个API分流挂载器
router = APIRouter()

#初始化模型
#之前全局只有一个 agent，所有用户共用一份记忆，会串话
#改成按 session_id 各存一个；懒加载：只有调用的时候才创建
_agents : Dict[str, RepoRouter] = {}
MAX_SESSIONS = 100   #防止内存无限涨


def get_agent(session_id : str)->RepoRouter:
    if session_id not in _agents:
        if len(_agents) >= MAX_SESSIONS:
            _agents.pop(next(iter(_agents)))   #删掉最早的会话
        _agents[session_id] = RepoRouter(max_history_turns=4)
    return _agents[session_id]

#==========数据模型============
#全部都封装起来
class ChatRequest(BaseModel):
    #BaseModel:方便尤其是字典数据传输
    #类型注解：工程化常用。保证类型合理
    #请求JSON格式标注。
    question :str
    session_id : str = Field(min_length=1,pattern=r"\S")

class ChatResponse(BaseModel):
    answer :str

class CodeAnalyseRequest(BaseModel):
    code :str
class DockerfileRequest(BaseModel):
    project_description : str
class LogAnalyseRequest(BaseModel):
    log_content :str


#=============路由=============



#聊天 :
#.post表示被发送的操作
#注意：agent.chat 是同步阻塞调用，用普通 def，FastAPI 会放到线程池里跑，不会卡住其他请求
@router.post("/chat",response_model = ChatResponse) #表明返回的一定包含ChatResponse里面的answer字段
            #表明输入的一定是从ChatRequest来的字段
def chat_endpoint(request : ChatRequest):
    """ 和 RepoRouter 对话 """
    try:
        agent = get_agent(request.session_id)
        answer = agent.chat(request.question)
        #返回值也封装成类
        return ChatResponse(answer=answer) #指定参数传值
    #工程中的try-except：如果运行主体出问题，不能直接崩溃报错吧，得切到except
        #将错误信息存到e中
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e)) #返回错误码和错误信息(转成字符串)


#代码检查
@router.post("/analyse-code")
#路由web代码用endpoint区分。可能调用里面不带endpoint的函数
#工程习惯问题
def analyse_code_endpoint(request : CodeAnalyseRequest):
    """代码分析：直接调用 AST 工具，不走大模型"""
            #一般都是返回字典？键+值清晰
            #相应的JSON格式。
    return {"analyse" : analyse_code(request.code)}

#docker-file生成接口
@router.post("/generate-dockerfile")
def generate_dockerfile_endpoint(request : DockerfileRequest):
    return {"docker-file": gen_docker_file(request.project_description)}



#日志分析接口
@router.post("/log-analyse")
def log_analyse_endpoint(request : LogAnalyseRequest):
    return {"log-analyse" : analyse_log(request.log_content)}

#清空记忆接口
@router.post("/clear-memory")
def clear_memory_endpoint(session_id : str = Query(min_length=1,pattern=r"\S")):
    #1.get_agent会返回已有，但不存在时会创建；清空时只查字典
    agent = _agents.get(session_id)
    if agent is not None:
        agent.clear_memory()
    return {"status" : "OK"}

#检查健康接口：运维会不断发送信息检查运行状态

@router.get("/health")
async def health():
    """API层健康检查"""
    return {"status" : "healthy"}
