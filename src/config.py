#项目配置文件：核心参数，重要配置数据。
#配置与代码分离
from pydantic_settings import BaseSettings,SettingsConfigDict


class Settings(BaseSettings):
    #每个变量都对应一个配置参数
    #继承父类BaseSettings:每个变量会自动到关键的.env文件里面去找

    #LLM配置
    deepseek_api_key : str#不设置默认值,.env里面必须有
    deepseek_base_url : str = "https://api.deepseek.com" #有默认值
    #deepseek-chat 已下线，换成 V4.1 Flash；对比实验用 deepseek-v4-pro
    model_name : str = "deepseek-flash"
    #V4 默认开思考模式。Agent 默认关掉：快、便宜
    llm_thinking : bool = False
    embedding_model : str = "text-embedding-ada-002"

    #向量数据库配置（新模型要重新建库，放到新目录，不和旧库混）
    Chroma_persist_dir : str = "./data/indices/src_current_20261025"

    #软编码。在config中指定配置
    bm25_persist_dir : str = "./data/indices/src_current_20261025/bm25_index.pkl"  #bm25只需要一个具体文件

    #模型选择
    embedding_provider : str = "qwen"
    #embedding_provider : str = "local"
    #模型名称：本地换成支持中文的 Qwen3-Embedding
    local_embedding_name : str = "Qwen/Qwen3-Embedding-0.6B"

    qwen_api_key : str
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_embedding_name : str =  "qwen3.7-text-embedding"
    qwen_rerank_name : str = "qwen3.7-text-rerank"

    #LLM 评委：用另一家的模型，避免“自己给自己打分”
    judge_model : str = "qwen3.8-max"

    # Gemini（google-genai，如 experiments/routing_data/generate_data.py）；不设必填以免未用 Gemini 时无法启动主程序
    google_api_key: str = ""
    #训练数据生成：用和被评测模型（DeepSeek、Qwen）不同家的最强模型，避免“自己出题自己考”
    data_gen_model : str = "gemini-4-argon"
    #填了就走 OpenAI 兼容接口（比如中转站）生成数据；不填就用 google_api_key 直连 Gemini
    data_gen_base_url : str = ""
    data_gen_api_key : str = ""

    #路由：native = DeepSeek 原生 function calling；phase2a = 自己微调的小模型派单
    routing_backend : str = "native"
    router_base_url : str = "http://127.0.0.1:8080/v1"   #llama.cpp / vLLM 的 OpenAI 兼容地址
    router_model : str = "repoagent-router"

    #rerank模型选择
    reranking_provider:str = "qwen"
    #reranking_provider:str = "local"
    #RAG配置
    #char模式下Python片段的最大字符数和重叠；AST固定阈值及回退规则见splitter.py
    code_chunk_size:int = 2000
    code_chunk_overlap:int = 350
    #代码切块方式：char = 按字符数切；ast = 按语法树切（对比实验见 scripts/build_knowledge_base.py --chunk-mode）
    code_chunk_mode : str = "char"
    #非Python文本在两种模式下都按这两个字符参数切分
    text_chunk_size : int = 1500
    text_chunk_overlap : int = 350
    #每个片段重叠字符数
      
    top_k : int = 8
    #是否启用rewriter模型调用      显式声明类型！
    enable_query_rewrite_in_tool : bool = False
    # 软熔断：Dense top-1 距离大于阈值则不再返回检索片段（仅 search_codebase）
    soft_fuse_enabled : bool = False
    soft_fuse_distance_threshold : float | None = None
    #隐藏文件指定 (旧版 BaseSettings 会去找 Config 嵌套类)
    #根目录找.env的隐藏文件。新版用 model_config 保存同样的设置。
    model_config = SettingsConfigDict(env_file=".env")

#创建全局唯一的Settings实例：单例模式
settings = Settings()
