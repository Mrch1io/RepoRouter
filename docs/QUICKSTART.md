# ⚡ RepoRouter 快速开始指南

> 按当前代码配置环境、建立索引并启动 RepoRouter。首次安装和建库耗时取决于网络与仓库大小。  
> 详细说明见 [README_CN.md](../README_CN.md)

---

## 📋 前置检查清单

- [ ] Python 3.12 已安装（项目依赖与测试使用的版本，`python --version`）
- [ ] pip 已安装（`python -m pip --version`）
- [ ] DeepSeek API 密钥（[获取](https://platform.deepseek.com)）
- [ ] Qwen API 密钥（[获取](https://dashscope.aliyuncs.com)）

> 默认使用 DeepSeek 回答、Qwen 嵌入，需要两家的有效密钥。`embedding_provider=local` 只替换嵌入模型，回答仍调用 DeepSeek；本地嵌入的安装与配置见下方 Q3。

---

## 🚀 快速启动

### 步骤 1️⃣：进入项目根目录

克隆或解压项目后，进入同时包含 `web_ui.py`、`requirements.txt` 和 `src/` 的目录。下面以项目目录 `RepoRouter` 为例；目录名不同时按实际路径替换，已在根目录则跳过此命令。后续命令都从项目根目录执行。

```bash
git clone https://github.com/Mrch1io/RepoRouter.git  #已下载项目则跳过
cd RepoRouter
```

### 步骤 2️⃣：创建虚拟环境并安装依赖

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS / Linux：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

如果 PowerShell 禁止执行激活脚本，可以不激活，后续把 `python` 换成 `.\.venv\Scripts\python.exe`，例如 `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`。默认 API 模式只需安装 `requirements.txt`；本地模型依赖见 Q3。

### 步骤 3️⃣：配置 API 密钥

首次配置时复制 `.env.example` 为 `.env`，再填入密钥。已有 `.env` 时直接检查配置，保留原文件。

Windows PowerShell：

```powershell
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
```

macOS / Linux：

```bash
if [ ! -e .env ]; then cp .env.example .env; fi
```

编辑 `.env`：

```env
# DeepSeek（LLM 主引擎，必需）
deepseek_api_key=sk-your-key-here
deepseek_base_url=https://api.deepseek.com
model_name=deepseek-flash
llm_thinking=False

# Qwen（默认嵌入方式，需要有效密钥）
qwen_api_key=sk-your-key-here
qwen_base_url=https://dashscope.aliyuncs.com/compatible-mode/v1
embedding_provider=qwen
qwen_embedding_name=qwen3.7-text-embedding

# 先用原生路由，无需启动额外的小模型服务
routing_backend=native
code_chunk_mode=char

# 为这次演示使用新索引目录
Chroma_persist_dir=./data/indices/demo
bm25_persist_dir=./data/indices/demo/bm25_index.pkl
```

> `.env` 已列入 `.gitignore`。配置加载要求 `deepseek_api_key` 和 `qwen_api_key` 两个字段都存在；默认 API 模式下都需填有效值。

### 步骤 4️⃣：构建知识库

确认 `data/indices/demo` 尚不存在，然后运行；Qwen 模式下建库会调用嵌入 API 并产生费用。

```bash
python scripts/build_knowledge_base.py --repo-path src --persist-dir data/indices/demo --chunk-mode char
```

**预期输出**：

```
正在从<源码绝对路径>建立本地知识库（embedding：qwen3.7-text-embedding）

本地知识库建立完成！保存在 <索引绝对路径>
```

这次生成的 Chroma 和 `bm25_index.pkl` 都位于 `data/indices/demo/`。**脚本只修改本次进程中的路径，不会自动写回 `.env`**，启动前确认 `.env` 中的两个索引路径与步骤 3 一致。

脚本拒绝覆盖任何已有目标目录，包括空目录或失败后留下的目录。需要重建时另取新目录，例如 `data/indices/demo_v2`，并同步修改 `.env` 中的两个路径；保留旧索引。源码、切块方式或嵌入模型改变后也需要重建。若用 `--embedding` 指定了其他 Qwen 模型，查询时的 `.env` 中 `qwen_embedding_name` 必须与之相同。

当前默认加载器会排除已知考题文件；这里建立的是当前源码的演示索引，不是历史评测快照。不要用它覆盖历史索引或据此声称复现了历史成绩。

### 步骤 5️⃣：启动 Web UI（推荐）

```bash
python -m streamlit run web_ui.py
```

访问 [http://localhost:8501](http://localhost:8501) 即可开始对话。网页直接调用 Agent，无需同时启动 FastAPI。网页启动时会创建 `RepoRouter` 并加载索引，所以先完成建库；修改 `.env` 后重启应用。

---

## 💬 第一次对话

在 Web UI 中输入：

```
请检索 vectorstore.py，解释 VectorStore.index_document 如何保存 Chroma 和 BM25 索引。
```

展开回答中的“处理详情”，检查是否调用 `search_codebase`、命中了哪些文件。工具选择由模型决定，回答可能有误，涉及实现细节时应核对返回的代码片段。

要试静态分析，请直接粘贴 Python 源码，例如：

```text
请用 analyse_code 检查这段代码：
def append_item(item, items=[]):
    items.append(item)
    return items
```

`analyse_code` 接收源码字符串，不会根据文件路径读取文件。

也可尝试（需要代码或日志的工具，请同时粘贴对应内容）：

- `找一下向量数据库的初始化函数` → 调用 `search_codebase`
- `帮我生成这段代码的单元测试` → 调用 `generate_tests`
- `这段报错怎么解决？`（粘贴日志）→ 调用 `analyse_log`

---

## 🔄 其他启动方式

### 方式 B：FastAPI 后端（用于集成）

```bash
# 在项目根目录执行
python -m uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

- API 文档：[http://localhost:8000/docs](http://localhost:8000/docs)
- 健康检查：`GET /health`，返回 `{"status":"OK"}`；这只验证服务已启动，不验证模型密钥或索引
- 聊天接口：`POST /api/v1/chat`

另开一个终端发送请求。Windows PowerShell：

```powershell
$body = @{ question = "找一下 vectorstore.py 中的 search 方法"; session_id = "demo-1" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/chat" -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

macOS / Linux：

```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"question":"找一下 vectorstore.py 中的 search 方法","session_id":"demo-1"}'
```

> 请求体需要 `question` 和非空、非纯空白的 `session_id`。同一对话使用同一个 ID，不同对话使用不同 ID；缺少 `session_id` 会返回 422。聊天请求需要有效密钥和已建好的索引。

### 方式 C：Python 脚本直接调用

```python
from src.agent.core import RepoRouter

agent = RepoRouter(max_history_turns=4)
response = agent.chat("请检索 core.py，解释 _prune_memory 如何裁剪对话历史。")
print(response)
```

在项目根目录保存脚本并运行：`python your_script.py`。构造 `RepoRouter` 时会加载 `.env` 指向的索引。

### 可选：切换到微调小模型路由

先按照 [训练与部署说明](../lora/autodl/README.md) 准备模型并启动 OpenAI 兼容的路由服务，再修改 `.env`：

```env
routing_backend=phase2a
router_base_url=http://127.0.0.1:8080/v1
router_model=repoagent-router
```

`router_model` 应与服务暴露的模型名或别名一致。切换后重启应用；小模型负责路由，回答仍调用 DeepSeek，代码检索仍需要步骤 4 的索引。

---

## ❓ 常见问题排查

### Q1: `ImportError: No module named 'src'`

**原因**：未在项目根目录运行，或 Python 路径不正确。

**解决**：

回到包含 `web_ui.py` 的项目根目录，使用 `python -m streamlit run web_ui.py` 启动网页，使用 `python -m pytest` 运行测试。`build_knowledge_base.py` 自行处理了导入路径，可按步骤 4 直接运行。

### Q2: API 连接超时

**排查**：下面通过 pytest 真正调用模型，会产生 API 费用，无需先建索引。

```bash
python -m pytest -m live tests/test_llm.py -q -s
```

若失败，检查 `.env` 中的 `deepseek_api_key`、`deepseek_base_url`、账号可用模型和网络连接。这个测试文件会分别调用 `deepseek-flash` 与 `deepseek-v4-pro`。

### Q3: Qwen 密钥缺失，或想使用本地嵌入

默认 `embedding_provider=qwen` 时，需要有效的 `qwen_api_key`。若选择本地嵌入，先在同一虚拟环境补装可选依赖：

```bash
python -m pip install langchain-huggingface sentence-transformers
```

再修改 `.env`：

```env
embedding_provider=local
local_embedding_name=Qwen/Qwen3-Embedding-0.6B
# 配置类仍要求该字段存在；不使用 Qwen 嵌入、重排和评委时可为空
qwen_api_key=
Chroma_persist_dir=./data/indices/demo_local
bm25_persist_dir=./data/indices/demo_local/bm25_index.pkl
```

首次使用会下载模型，需要网络及相应磁盘、内存空间。保留有效的 `deepseek_api_key`，并在新的、尚不存在的目录建库：

```bash
python scripts/build_knowledge_base.py --repo-path src --persist-dir data/indices/demo_local --chunk-mode char
```

本地与 Qwen 嵌入不能混用同一索引。主流程不需要重排或评委；若另外启用 Qwen 重排或评委，仍需有效的 Qwen 密钥。

### Q4: 内存不足 `MemoryError`

**解决**：关闭占用内存的程序；若使用本地嵌入，可按步骤 3 改回 Qwen API 并另建索引；大仓库可以用 `--repo-path` 指定实际需要的源码子目录。

仅修改 `.env` 的 `top_k` 不会改变当前在线工具的返回数量：`search_codebase` 使用调用参数 `top_k`，未传时为 6。返回片段数也不决定整个索引和本地模型加载时的内存占用。

### Q5: 响应很慢（>5 秒）

**改进**：

- 查看网页总耗时及“处理详情”中的工具调用记录，并结合终端的加载提示和函数耗时定位问题。
- 检查 `.env` 的 `llm_thinking=False`、`enable_query_rewrite_in_tool=False`；开启这些功能会增加推理或模型调用开销。
- 使用 `phase2a` 时，另检查路由服务是否可访问。应用没有固定的 5 秒响应保证。

---

## ✅ 验证安装成功

| 检查项 | 命令 | 预期 |
|--------|------|------|
| 配置加载 | `python -m tests.test_config` | 显示 DeepSeek key 是否非空、模型名，不打印密钥；不验证密钥有效性 |
| 离线回归 | `python -m pytest -q` | 默认排除 `live` 用例，不调用真实模型 API |
| LLM 连接（会付费） | `python -m pytest -m live tests/test_llm.py -q -s` | 真正运行两种模型的连接测试 |
| 工具调用（会付费） | `python -m pytest -m live tests/test_tool_calls.py -q -s` | 使用当前索引与模型验证检索和路由 |

离线测试也会加载配置，先完成步骤 3；只跑离线测试时两个密钥字段可用占位值。工具调用测试包含针对本项目 `vectorstore.py` 的断言，运行前应使用步骤 4 建立的 `src/` 索引，并设 `routing_backend=native`。真实模型输出有波动，测试失败需要结合 trace 排查。

---

## 📁 项目结构速览

```
RepoRouter/
├── web_ui.py              ← Streamlit UI（入口）
├── .env.example           ← 配置模板，复制为 .env
├── src/
│   ├── main.py            ← FastAPI 应用
│   ├── config.py          ← 配置管理
│   ├── api/               ← HTTP 接口
│   ├── agent/
│   │   ├── core.py        ← Agent 核心
│   │   ├── router.py      ← 小模型路由
│   │   └── tools/         ← 6 个工具
│   ├── rag/               ← 向量库、BM25、Query Rewriter
│   └── evaluation/        ← 检索、回答评测与统计
├── scripts/               ← 建库、评测、审计与绘图入口
│   └── build_knowledge_base.py
├── experiments/routing_data/ ← 训练题生成、清洗与离线重建
├── lora/                  ← 训练数据与路由评测
│   └── autodl/            ← 训练与部署配置
├── data/
│   ├── v2/                ← 第二版原始结果与日志
│   ├── reports/           ← 历史评测汇总
│   └── indices/demo/      ← 本指南新建的 Chroma 与 bm25_index.pkl 目录
├── 评估/第二版/           ← 按实验主题整理的报告
├── archive/early-experiments/ ← 早期实验归档
├── docs/                  ← 使用、架构说明及展示图片
├── saves/                 ← 本地模型与训练记录，已列入忽略规则
└── tests/                 ← 回归测试，默认跳过真实服务
```

---

## 🎓 下一步学习

| 想要... | 阅读 |
|--------|------|
| 理解当前实现 | [对话循环](../src/agent/core.py) · [检索工具](../src/agent/tools/search_tool.py) |
| 添加新工具 | [工具菜单与参数校验](../src/agent/tools/schemas.py) · [工具注册](../src/agent/core.py) |
| 小模型训练与部署 | [AutoDL 操作说明](../lora/autodl/README.md) |
| 评估与测试 | [README_CN.md#评测设计](../README_CN.md#评测设计每套题只回答一个问题) |

---

## 💡 实用提示

### 查看工具耗时

`chat` 和 `search_codebase` 会打印耗时，例如 `search_codebase` 的输出：

```
函数 'search_codebase' 执行完毕 | 耗时: 0.234 秒
```

---

## ✨ 你现在可以做什么

1. ✅ 在 Web UI 中提问
2. ✅ 搜索代码库（`search_codebase`）
3. ✅ 分析代码质量（`analyse_code`）
4. ✅ 生成单元测试（`generate_tests`）
5. ✅ 分析错误日志（`analyse_log`）
6. ✅ 生成 Docker 配置（`generate_dockerfile`）
7. ✅ 智能代码补全（`complete_code`）

---
