# 📐 RepoRouter 系统架构设计文档

> 本文档说明 RepoRouter 的系统设计、技术选型和核心算法。结构图和标注为“节选”的代码用于对照源码；可运行示例另行标明。  
> 快速上手请参考 [README_CN.md](../README_CN.md) 与 [QUICKSTART.md](./QUICKSTART.md)。

---

## 目录
1. [整体架构](#整体架构)
2. [核心模块设计](#核心模块设计)
3. [RAG 系统详解](#rag-系统详解)
4. [Agent 决策引擎](#agent-决策引擎)
5. [数据流与状态管理](#数据流与状态管理)
6. [性能优化策略](#性能优化策略)
7. [扩展与定制](#扩展与定制)

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         Presentation Layer                       │
│  ┌────────────────────┐             ┌────────────────────────┐  │
│  │ Streamlit Web UI   │             │ FastAPI REST API       │  │
│  │ (web_ui.py)        │             │ (/api/v1/chat)         │  │
│  └────────────────────┘             └────────────────────────┘  │
│                                              ▲                    │
└──────────────────────────────────────────────┼────────────────────┘
                                               │
┌──────────────────────────────────────────────┼────────────────────┐
│                  Application Layer                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              RepoRouter Core Engine                         │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ Multi-turn Dialogue Manager                         │  │  │
│  │  │ - Memory Buffer: max_history_turns                  │  │  │
│  │  │ - Context Window Management                         │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │              Agent Decision Engine                        │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │ Tool Router (LLM Function Calling)                  │  │  │
│  │  │ - 解析 tool_calls 中的 name / arguments              │  │  │
│  │  │ - 验证工具名、参数，执行并反馈                        │  │  │
│  │  │ - 错误恢复（safe_json_loads 解析 arguments）          │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   ┌─────────┐    ┌────────┐    ┌──────────┐
   │Tools    │    │LLM     │    │RAG       │
   │Layer    │    │Layer   │    │System    │
   └─────────┘    └────────┘    └──────────┘
```

### 架构分层说明

网页与 API 分别直接调用 `RepoRouter`，网页不经过 HTTP 调用 API。网页在 Streamlit 会话中保存 Agent；API 按 `session_id` 保存 Agent。

| 层级 | 组件 | 职责 |
|------|------|------|
| **Presentation** | web_ui.py / FastAPI | 用户交互、API 暴露 |
| **Application** | Agent Core | 业务编排、工具路由、记忆管理 |
| **Service** | Tools / LLM / RAG | 具体能力实现 |
| **Infrastructure** | config.py / .env | 配置管理（Pydantic Settings） |

---

## 核心模块设计

### 1️⃣ Agent 核心引擎 (`src/agent/core.py`)

#### 类图
```text
class RepoRouter:
    - available_tools: Dict[str, Callable]  # 工具映射表（6 个工具）
    - memory: List[Dict]                     # 多轮对话历史（含 system/user/assistant/tool）
    - max_history_turns: int                 # 记忆窗口大小
    
    + __init__(max_history_turns=1)
    + chat(question: str) -> str             # 主对话入口
    - _prune_memory()                        # 记忆修剪（过滤 tool_calls 残余）
```

#### 核心工作流

`settings.routing_backend` 选择两条路径：

- `native`：DeepSeek 接收历史和 `TOOLS_SCHEMA`，最多循环 5 次；`_run_tool` 校验工具名及参数，同一轮阻止重复调用。工具结果反馈后继续生成回答。
- `phase2a`：小模型接收当前问题和保留的历史，输出 `tool_call`、`no_tool` 或 `clarify_needed`。需要工具时执行一次，再由大模型写答案；追问则直接返回。运行时仓库简介来自当前检索索引，历史评测仍保留独立提示词快照。

两条路径都在成功后记录最终回复并裁剪历史；原生模式达到迭代上限时也记录停止回复。异常时恢复本轮开始前的历史，错误交给调用方处理。以下是外层回滚逻辑的节选，具体分支见 [`core.py`](../src/agent/core.py)：

```python
def chat(self,question : str,trace : list | None = None,max_tokens : int = 800)->str:
    #1.已有消息内容不变，只复制列表
    history = list(self.memory)
    try:
        return self._chat(question,trace,max_tokens)
    except Exception:
        #2.失败退回本轮开始前，异常照常抛给调用方
        self.memory = history
        raise
```

### 2️⃣ LLM 接口层 (`src/agent/llm.py`)

#### 设计原则
- **单一职责**：只负责 LLM 通信，不处理业务逻辑
- **兼容性**：使用 OpenAI 标准接口，支持任何兼容 API
- **可观察性**：记录 token 用量、估算费用及回答结束原因；并非完整请求/响应审计日志

下面是接口节选，省略了模型选择、thinking 参数和用量累计，完整实现以 [`llm.py`](../src/agent/llm.py) 为准。

```python
def chat_completion(
    messages: List[Dict],
    max_tokens: int = 800,
    temperature: float = 0.2,
    tools: Optional[List] = None
):
    """
    发送请求到 LLM（OpenAI 兼容接口，默认 DeepSeek）
    
    Args:
        messages: 对话历史（含 system/user/assistant/tool）
        tools: 工具定义（OpenAI Function Calling 格式）
    
    Returns:
        response.choices[0].message（含 content、tool_calls 等）
    """
    response = client.chat.completions.create(
        model=settings.model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        tools=tools
    )
    return response.choices[0].message
```

#### 关键配置
```python
# 模型参数
model_name = "deepseek-flash"      #当前配置默认值
temperature = 0.2                  #降低采样随机性，不保证完全确定
max_tokens = 800                   #单次生成上限，调用方可调整

# API 参数
base_url = "https://api.deepseek.com"
timeout = 120                      #当前SDK客户端的超时设置
```

### 3️⃣ 工具系统 (`src/agent/tools/`)

> 早期多工具串联（Tool Chaining）示例在[早期 06 号报告](../archive/early-experiments/评估/06-多工具协作.md)里。

#### 工具架构

```
src/agent/tools/
  ├── schemas.py      ← OpenAI Function Calling 格式 Schema
  ├── code_analyser.py      ← AST 静态分析
  ├── search_tool.py        ← RAG 代码检索
  ├── code_completion.py  ← 代码补全
  ├── dockerfile.py    ← Dockerfile 生成
  ├── test_generation.py← 单元测试生成
  └── log_analysis.py   ← 日志分析
```

#### 标准工具接口

工具接收 Schema 声明的参数，返回字符串；校验与错误记录由 `RepoRouter._run_tool` 统一处理。下面直接调用现有静态分析工具，可在项目根目录运行，不需要模型或索引：

```python
from src.agent.tools.code_analyser import analyse_code

result = analyse_code("def collect(items=[]):\n    return items\n")
assert "函数'collect'使用了默认可变参数 []" in result
print(result)
```

#### 工具 Schema 示例（OpenAI Function Calling 格式）

```python
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_codebase",
            "description": "在本地代码库中检索相关代码片段...",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词/短句"},
                    "top_k": {"type": "integer", "description": "返回片段数量", "default": 6}
                },
                "required": ["query"]
            }
        }
    },
    # ... 其他 5 个工具 ...
]
```

> ⚠️ **注意**：Schema 中的 `function.name` 必须与 `available_tools` 中的键完全一致。

---

## RAG 系统详解

> 📊 **早期评估结论**（[原始报告](../archive/early-experiments/评估/README.md)）：基于 01-04 号实验，**默认纯 Dense**，不启用 Reranker、Hybrid；Query Rewrite 对口语化 query 有显著提升（MRR +16.5%）。

### 整体流程

```
用户输入
  ↓
Query Rewriter (可选，enable_query_rewrite_in_tool)
  ↓
┌─ 密集检索 (Dense Retrieval) ← 主链路
│   ├─ 嵌入模型: settings.qwen_embedding_name
│   ├─ 向量库: Chroma
│   └─ 返回: Top-K 向量相似结果
│
├─ 稀疏检索 (Sparse Retrieval)
│   ├─ 方法: BM25 (词频-逆文档频率)
│   ├─ 存储: Pickle 序列化
│   └─ 返回: Top-K 关键词匹配结果
│
→ 融合 (Fusion)
  ├─ 策略: RRF (Reciprocal Rank Fusion)
  └─ 返回: 融合后的 Top-K
│
→ 重排 (Re-ranking，可选)
  ├─ 模型: CrossEncoder
  ├─ 作用: 精排混合结果
  └─ 返回: 最终 Top-K
  
  
↓
向 LLM 注入为 Context
```

**工程决策**（来自[早期 03 号重排消融实验](../archive/early-experiments/评估/03-重排消融.md)）：
- 纯 Dense MRR 已达 **0.944**，叠加 Reranker 反而下降
- BM25 在代码中 def/import/for 等高频词导致噪声
- 主链路默认纯 Dense；总延迟还包含嵌入 API、模型生成和可选改写，不能用局部检索耗时代替端到端耗时

### 1️⃣ 向量存储 (`src/rag/vectorstore.py`)

> 早期实现摘要在[早期 07 号报告](../archive/early-experiments/评估/07-向量库实现.md)里。

#### 设计

以下为结构节选，省略维度设置、重排器初始化、索引检查及简介缓存；正式建库入口还会拒绝已有目标目录，避免覆盖历史索引。

```python
class VectorStore:
    """
    Chroma 向量数据库 + BM25 的封装，支持 Dense / Hybrid / 重排
    """
    
    def __init__(self, use_reranker: bool = False):
        # 嵌入模型：qwen（API）/ local（HuggingFace）
        if settings.embedding_provider == "qwen":
            self.embeddings = OpenAIEmbeddings(
                model=settings.qwen_embedding_name,
                openai_api_key=settings.qwen_api_key,
                openai_api_base=settings.qwen_base_url
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(model=settings.local_embedding_name)
        
        self._db = None
        self.bm25_retriever = BM25Retriever()
        
    def index_document(self, all_chunks: List[Document]):
        """构建索引（scripts/build_knowledge_base.py 调用）"""
        self._db = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory=settings.Chroma_persist_dir
        )
        self.bm25_retriever.index_document(all_chunks)
        
    def load(self):
        """加载已有索引（search_tool 单例调用）"""
        self._db = Chroma(
            embedding_function=self.embeddings,
            persist_directory=settings.Chroma_persist_dir
        )
        self.bm25_retriever.load()
```

#### 检索方法

`search_codebase` 工具调用 `search_dense_with_top1`，一次召回同时返回片段和粗排 top-1 距离，供软熔断判断。`search_dense` 是只取片段的入口；`search_hybrid` 和 `search_bm25` 用于对比实验。是否重排由 `VectorStore(use_reranker=...)` 决定。

混合路径先从 Dense 和 BM25 各召回至少 10 个候选，再用 RRF 合并；启用重排时，`search_hybrid` 对候选进一步精排。RRF 分数为 `Σ 1 / (60 + rank)`，排名从 1 开始。同一路先去重，重复块不占后续名次；块身份由来源、块号、行号和正文共同决定，不能只按文件名合并。

下面复用现有 RRF 实现，可离线运行。`object.__new__` 仅用于这个算法示例，跳过嵌入客户端初始化；正式检索仍需正常创建并加载 `VectorStore`。

```python
from langchain_core.documents import Document
from src.rag.vectorstore import VectorStore

#1.准备同文件的两个不同代码块
first = Document(page_content="def first(): pass",metadata={"source":"a.py","chunk_id":0})
second = Document(page_content="def second(): pass",metadata={"source":"a.py","chunk_id":1})
store = object.__new__(VectorStore)

#2.第一路重复的first只计一次，第二路为second增加一份分数
result = store._rrf_fusion([first,first,second],[second],top_k=2)
assert result == [second,first]
assert len({store._chunk_key(doc) for doc in result}) == 2
```

### 2️⃣ BM25 稀疏检索 (`src/rag/bm25.py`)

#### BM25 算法原理

```
BM25 得分 = IDF(q_i) * (f(q_i, D) * (k1 + 1)) / (f(q_i, D) + k1 * (1 - b + b * |D| / avgdl))

其中：
- q_i: 查询中的第 i 个词
- f(q_i, D): 词 q_i 在文档 D 中的频率
- |D|: 文档 D 的长度
- avgdl: 平均文档长度
- k1, b: 参数（通常 k1=2.0, b=0.75）
- IDF: 逆文档频率 = log((N - n(q_i) + 0.5) / (n(q_i) + 0.5))
```

#### 实现

索引的读写与排序复用 [`BM25Retriever`](../src/rag/bm25.py)。当前版本按完整标识符和单个汉字分词，只对命中查询词的块排序，用 `nlargest` 取前 K 个；旧版本索引会在内存中重新分词，不改写原文件。下面是可离线运行的分词示例：

```python
from src.rag.bm25 import _tokenize

assert _tokenize("中文 search_codebase") == ["中","文","search_codebase"]
```

### 3️⃣ Query Rewriter (`src/rag/query_rewriter.py`)

#### 设计目标

改写用户的**口语化、模糊查询**为**信息密度高、精确的检索词组合**。

#### 改写策略

```
原始查询：
"妈妈，请帮我找一下切分文本的函数定义def，呜呜呜。。。。。"
  ↓ [降噪]
"找切分文本的函数定义def"
  ↓ [实体保留]
保留：切分、文本、函数、def
  ↓ [术语扩展]
添加：TextSplitter、Document、Chunk 等专业术语
  ↓ [改写后]
"切分文本 TextSplitter Document Chunk 函数定义 def"
```

#### 实现

下面调用现有实现，需要模型服务。`chat_completion` 返回 message 对象，解析前要取 `.content`；当前实现还会检查 `dense_query` 是否为非空字符串，解析失败则记录错误并回退原问题。

```python
from src.rag.query_rewriter import Query_rewriter

result = Query_rewriter().rewrite("CodeSplitter 怎么切 Python 代码？")
print(result.dense_query)
```

---

## Agent 决策引擎

### 工具路由决策流程（Function Calling）

```
LLM 返回 message（含 tool_calls）
  ↓
┌──────────────────────────────────┐
│ 解析 tool_calls                  │
│ - function["name"] → 工具名       │
│ - function["arguments"] → JSON 字符串 │
│ - safe_json_loads(arguments) → 参数字典 │
└──────────────────────────────────┘
  ↓
┌──────────────────────────────────┐
│ 工具名验证                       │
│ - 检查是否在 available_tools 中   │
│ - 处理无效工具（返回错误）        │
└──────────────────────────────────┘
  ↓
┌──────────────────────────────────┐
│ 参数验证                         │
│ - 类型检查                       │
│ - 范围检查                       │
└──────────────────────────────────┘
  ↓
┌──────────────────────────────────┐
│ 工具执行                         │
│ - try/catch 错误处理              │
│ - 结果格式化                     │
└──────────────────────────────────┘
  ↓
反馈给 LLM 进行总结
```

### JSON 解析的鲁棒性处理

```python
def safe_json_loads(s: str) -> dict:
    """
    鲁棒的 JSON 解析，处理常见格式错误
    """
    if not isinstance(s, str):
        return {}
    
    s = s.strip()
    
    # 移除 ```json``` 标记
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    
    # 提取 { ... } 内容
    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        s = s[l:r+1]
    
    try:
        return json.loads(s)
    except Exception as e:
        print(f"⚠️ JSON 解析失败: {e}")
        return {}
```

---

## 数据流与状态管理

### 单轮对话的完整数据流

下图对应 `native` 分支；`phase2a` 的分流见前面的核心工作流。

```
输入: query (用户问题)
  ↓
[1] Query Preprocessing
    - 接收调用方传入的问题
    - 保存本轮开始前的历史，供异常回滚
  ↓
[2] Context Assembly
    - 从 self.memory 构建对话历史
    - 保存本轮用户提问（首轮先加入 system）
    - 随模型请求传入 TOOLS_SCHEMA
  ↓
[3] LLM Call - Tool Selection（循环内）
    LLM 输入:
    - 系统提示词 + 对话历史 + 工具定义 (TOOLS_SCHEMA)
    ↓
    LLM 输出 (message):
    - tool_calls: [{function: {name, arguments}, id}, ...]
    或
    - content: 最终回答（无工具调用时）
  ↓
[4] Tool Execution (if tool_calls)
    - safe_json_loads(arguments) 解析参数
    - available_tools.get(name) 获取工具
    - 执行并追加 role: "tool" 到 memory
  ↓
[5] 继续循环 或 返回
    - 有 tool_calls → 继续下一轮 LLM 调用
    - 无 tool_calls → 提取 content，_prune_memory()，返回
  ↓
[6] Memory Update
    - 保留 system 和最终用户/助手问答
    - 去除中间工具调用与工具结果
    - 按 max_history_turns 保留最近问答，不是按 token 截断
  ↓
输出: response (最终回答)
```

### 状态管理关键点

- 每个 Agent 实例有独立 `memory`，默认保留最近一轮完整问答。
- `_prune_memory` 先过滤 `tool` 和带 `tool_calls` 的中间消息，再保留最近 `2 * max_history_turns` 条用户/助手消息；system 单独保留。
- 不能直接截取原始消息列表的最后 N 条，否则可能留下不配对的工具结果或回答。
- 网页只有请求成功才保存本轮问答；失败时显示提示，后端同时回滚本轮。

---

## 性能优化策略

### 1️⃣ 向量库预热（核心优化，效果显著）

在 `RepoRouter.__init__` 中**提前触发**向量库单例加载，将 Chroma + BM25 的磁盘读取成本前置到 Agent 创建时。它移动了加载时机，没有消除这部分耗时。

以下为构造函数节选：

```python
# src/agent/core.py

class RepoRouter:
    def __init__(self, max_history_turns=1):
        self.available_tools = {...}
        self.memory = []
        self.max_history_turns = max_history_turns

        # 🔥 预热数据库：提前触发单例加载
        from src.agent.tools.search_tool import _get_vector_store
        _get_vector_store()  # 首次调用时加载 Chroma + BM25，后续复用
```

**加载时机对比**：

| 场景 | 首次 search_codebase 耗时 | 说明 |
|------|---------------------------|------|
| **无预热** | 首次调用时需加载 + 检索，延迟叠加 | 用户首轮对话体验差 |
| **有预热** | 加载在 Agent 创建时完成，检索时直接命中内存 | 首轮对话时向量库已就绪 |

历史日志输出示例（保留当时的路径与耗时）：当前默认 Chroma 目录已改为 `./data/indices/src`，BM25 索引为 `./data/indices/src/bm25_index.pkl`；以下日志中的旧路径不再使用。

**无预热**：先出系统提示词(模型初次被调用)，才加载向量库

加载系统提示词
Agent综合分析中
Agent决策：决定调用工具：search_codebase
[System] 首次加载向量库...
成功从路径./data/chorma中读取Chroma向量数据库
成功从硬盘持久路径读取到bm25索引引擎！
成功从路径data\bm25_index.pkl中读取bm25向量数据库

⏱️ [耗时监控] 函数 'search_codebase' 执行完毕 | 耗时: 7.452 秒


**有预热**：在被调用前加载向量库
```
[System] 首次加载向量库...
成功从路径./data/chorma中读取Chroma向量数据库
成功从硬盘持久路径读取到bm25索引引擎！
加载系统提示词
Agent综合分析中

Agent决策：决定调用工具：search_codebase
⏱️ [耗时监控] 函数 'search_codebase' 执行完毕 | 耗时: 3.365 秒
```

### 2️⃣ 向量库单例模式

以下节选包含初始化锁：只有加载成功才发布实例，避免并发请求拿到尚未加载或加载失败的对象。

```python
# src/agent/tools/search_tool.py

_vector_store: Optional[VectorStore] = None

def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        with _vector_store_lock:
            if _vector_store is None:
                store = VectorStore()
                store.load()
                _vector_store = store
    return _vector_store
```

### 3️⃣ 耗时监控（time_logger）

`search_codebase`、`chat` 等关键函数使用 `@time_logger` 装饰器，自动打印执行耗时，便于定位性能瓶颈：

```text
#历史输出示例
⏱️ [耗时监控] 函数 'search_codebase' 执行完毕 | 耗时: 3.365 秒
⏱️ [耗时监控] 函数 'chat' 执行完毕 | 耗时: 42.761 秒
```

### 4️⃣ 网络请求优化

```python
# src/agent/llm.py - 使用 OpenAI 官方 SDK（内置连接池和重试）
client = OpenAI(
    api_key=settings.deepseek_api_key,
    base_url=settings.deepseek_base_url
)

# 参数调优
temperature=0.2   #降低采样随机性，不代表更快收敛
max_tokens=800    # 适度长度 → 平衡质量和速度
```

### 5️⃣ 检索策略

- **Dense-only**：在评估集上 Hit@8=100%、MRR≈0.94，是当前主力检索引擎（见 [README_CN：混合检索真的比纯向量检索好吗](../README_CN.md#3-混合检索真的比纯向量检索好吗)）
- **RRF 融合**：Hybrid 模式下 Dense + BM25 采用 Reciprocal Rank Fusion
- **可选重排**：CrossEncoder / Qwen gte-rank-v2 精排

### 6️⃣ 文本处理优化

切块参数的作用范围如下，见 [`splitter.py`](../src/rag/splitter.py)：

| 文件与模式 | 大小和重叠参数 | 说明 |
| --- | --- | --- |
| Python，`char` | `code_chunk_size`、`code_chunk_overlap` | 默认 2000、350，按字符切分 |
| Python，`ast` | `max_chunk_size=1500`，保留实验设置 | 正常 AST 切块不使用上述字符参数；本地 astchunk 0.1.0 划窗按 UTF-8 字节计数并忽略 ASCII 空白，不能理解为输出最多 1500 字符 |
| 其他支持的文本 | `text_chunk_size`、`text_chunk_overlap` | 默认 1500、350，与 Python 的切块模式无关 |

AST 模式没有设置相邻正文重叠，`chunk_expansion` 补的是祖先信息。项目只保留简短的类名、函数名作为上下文，行号指向原始正文；补充的上下文也会增加输出长度。

当前依赖处理超长单行时可能漏掉正文。切块后会去除 ASCII 空白，按顺序核对原文和各块正文；若不一致，记录警告并用现有字符分割器回退整个文件。此时 `code_chunk_size`、`code_chunk_overlap` 生效，块保留来源和块号，但没有 AST 行号及祖先上下文。字符分割器本身会处理边界空白，因此这里不声称逐字节无损。

```python
# src/rag/splitter.py - 使用 LangChain 工业级分割器
self.py_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=2000,
    chunk_overlap=350
)
```

---

## 扩展与定制

### 1️⃣ 添加新工具

#### 步骤 1: 实现工具函数

这是扩展示例，不会自动创建文件。工具依赖 `llm.py`，不要反向导入会注册工具的 `core.py`。执行工具会调用模型，离线测试应替换 `chat_completion`。

```python
#src/agent/tools/my_new_tool.py

from src.agent.llm import chat_completion

def analyze_requirements(repo_structure : str)->str:
    """
    分析项目需求和依赖关系
    """
    SYSTEM_PROMPT = """
    你是一个项目需求分析专家...
    """
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"项目结构：\n{repo_structure}"}
    ]
    
    #1.SDK返回message对象，工具返回其中的正文
    return chat_completion(messages, max_tokens=1000).content or ""
```

#### 步骤 2: 注册到 Agent

在 `core.py` 顶部添加导入，并在现有 `RepoRouter.__init__` 的 `available_tools` 中增加一项；保留原来的初始化代码。

```python
#src/agent/core.py
from src.agent.tools.my_new_tool import analyze_requirements

#现有self.available_tools字典中新增：
#"analyze_requirements": analyze_requirements,
```

#### 步骤 3: 添加 Schema 定义

```python
# src/agent/tools/schemas.py

TOOLS_SCHEMA.append({
    "type": "function",
    "function": {
        "name": "analyze_requirements",  #必须与 available_tools 键一致
        "description": "分析项目的需求和依赖关系",
        "parameters": {
            "type": "object",
            "properties": {
                "repo_structure": {"type": "string", "description": "项目目录结构或源代码"}
            },
            "required": ["repo_structure"]
        }
    }
})
```

`check_tool_args` 会读取这份 Schema，无需再写一套校验。现有工具名为 `generate_dockerfile` 和 `generate_tests`，不能写成旧的 `generate_docker_file`、`generate_test`。如果使用 `phase2a`，还需单独更新当前路由的工具提示并验证模型能派发新工具；只改原生工具表不会更新已训练模型。历史提示词快照和评测结果应保留。

### 2️⃣ 更换 LLM

```python
# src/agent/llm.py

#方案A：修改.env，填入所用服务的OpenAI兼容地址、模型和对应密钥
#deepseek_base_url = 服务提供的完整兼容地址
#model_name = 该服务支持的模型名称
#deepseek_api_key = 该服务的密钥

# 方案 B：使用其他 SDK（如 Anthropic）
# 需适配返回格式，确保 tool_calls 结构兼容
```

### 3️⃣ 支持更多编程语言

以下只是扩展初始化示意。还需在 `split()` 中按后缀调用对应分割器；当前非 Python 文件使用文本分割器，AST 模式只处理 Python。

```python
# src/rag/splitter.py

class CodeSplitter:
    def __init__(self):
        # Python
        self.py_splitter = RecursiveCharacterTextSplitter.from_language(
            Language.PYTHON, chunk_size=2000, chunk_overlap=350
        )
        # Java
        self.java_splitter = RecursiveCharacterTextSplitter.from_language(
            Language.JAVA, chunk_size=2000, chunk_overlap=350
        )
        # Go
        self.go_splitter = RecursiveCharacterTextSplitter.from_language(
            Language.GO, chunk_size=2000, chunk_overlap=350
        )
        # ... 等等
```

---

## 部署与运维

### Docker 部署

下面是 API 镜像示例。先在宿主机为源码建立独立索引，运行时挂载索引目录和环境变量；构建镜像时不调用嵌入服务，也不把 `.env` 复制进镜像。Web UI 需另行添加其入口与资源。

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY lora/*.py ./lora/

# 启动 API（需在项目根目录，或设置 PYTHONPATH）
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

例如在项目根目录执行 `docker run --env-file .env -p 8000:8000 -v "${PWD}/data/indices:/app/data/indices" <镜像名>`。挂载路径需与 `Chroma_persist_dir`、`bm25_persist_dir` 对应；`phase2a` 还需让容器能访问路由服务。

### 监控与日志

```python
# 时间追踪装饰器
@time_logger
def search_codebase(query: str, top_k: int = 6) -> str:
    """
    自动记录执行时间：
    ⏱️ [耗时监控] 函数 'search_codebase' 执行完毕 | 耗时: 0.234 秒
    """
    ...
```

---

## 参考资源

- **RAG 论文**: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- **LoRA 论文**: [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- **BM25**: [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
- **RRF**: [Reciprocal Rank Fusion](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.36.8814)
