"""
生成 **Phase 2A — 单轮工具路由** 对齐样本（JSONL），**仅调用 Gemini**。

监督目标见 `ROUTING_TASK.md`：只优化路由决策（JSON 可解析、decision_type、tool、参数），
**不**以长篇自然语言解答为训练目标。每条使用 `decision_type` ∈ tool_call | no_tool | clarify_needed，
`clarify_answer` 仅用于 clarify_needed（短句澄清），与 `direct_answer` 脱钩。

切片与条数：三切片按 **权重** 缩放到 `TARGET_TOTAL`（默认权重对应旧版 500 条全量 300:100:100）。
  - private_domain → bucket=`domain`
  - dev_general_no_rag → bucket=`far_general`
  - misleading_edge → bucket=`misleading_edge`

依赖：`.env` 中 `google_api_key`（见 src/config.py），`pip install google-genai`。

用法：仓库根目录  
`python experiments/routing_data/generate_data.py`  
`python experiments/routing_data/generate_data.py --target 450 --segment-ratio 70,15,15 --split-test 50`  
`--dry-run` 只打印计划。

输出：`experiments/routing_data/outputs/ft_qa_{TARGET_TOTAL}.jsonl`；若 `--split-test N` 则另写  
`ft_qa_{TARGET_TOTAL}_train.jsonl` 与 `ft_qa_{TARGET_TOTAL}_test.jsonl`（分层抽样，默认可复现 seed）。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.routing_data.project_context import (
    PROJECT_CONTEXT_INTERNAL,
    SCENARIOS,
)
from experiments.routing_data.clean_data import collect_issues

BATCH_CHUNK = 20

# 默认生成条数（CLI `--target` 可覆盖）
TARGET_TOTAL = 100

# 三切片权重（与 TARGET_TOTAL 成比例缩放）；旧版 500 全量 = 300:100:100；扩量可用 70:15:15
DEFAULT_SEGMENT_WEIGHTS: tuple[int, int, int] = (300, 100, 100)

SLICE_ORDER: tuple[str, str, str] = (
    "private_domain",
    "dev_general_no_rag",
    "misleading_edge",
)

# 切片 → 对外 bucket 名（与训练/DPO 脚本对齐）
SLICE_TO_BUCKET: dict[str, str] = {
    "private_domain": "domain",
    "dev_general_no_rag": "far_general",
    "misleading_edge": "misleading_edge",
}

# 各切片内 decision_type 目标占比（单批内按整数条数分配，总和 = 本批 n）
# Phase 2A：tool_call 为主；far_general 提高 no_tool；misleading 提高 clarify_needed
MIX_BY_SLICE: dict[str, dict[str, float]] = {
    "private_domain": {"tool_call": 0.70, "no_tool": 0.15, "clarify_needed": 0.15},
    "dev_general_no_rag": {"tool_call": 0.35, "no_tool": 0.40, "clarify_needed": 0.25},
    "misleading_edge": {"tool_call": 0.40, "no_tool": 0.15, "clarify_needed": 0.45},
}


def _allocate_kind_counts(n: int, fractions: dict[str, float]) -> dict[str, int]:
    """将占比变为整数条数，保证总和为 n。"""
    keys = ("tool_call", "no_tool", "clarify_needed")
    raw = [fractions[k] * n for k in keys]
    base = [int(x) for x in raw]
    rem = n - sum(base)
    order = sorted(
        range(len(keys)),
        key=lambda i: (raw[i] - base[i], i),
        reverse=True,
    )
    for i in range(rem):
        base[order[i]] += 1
    return dict(zip(keys, base))


def _batch_sizes(n: int) -> list[int]:
    out: list[int] = []
    rest = n
    while rest >= BATCH_CHUNK:
        out.append(BATCH_CHUNK)
        rest -= BATCH_CHUNK
    if rest > 0:
        out.append(rest)
    return out


def _scaled_segment_totals(
    target_total: int,
    weights: tuple[int, int, int],
) -> list[tuple[str, int]]:
    """按 weights 比例缩放到 target_total，条数之和严格等于 target_total。"""
    if target_total <= 0:
        raise ValueError("target_total 必须为正整数")
    if len(weights) != 3 or any(w < 0 for w in weights):
        raise ValueError("weights 须为三个非负整数")
    base_sum = sum(weights)
    if base_sum == 0:
        raise ValueError("weights 之和不能为 0")
    if target_total == base_sum:
        return list(zip(SLICE_ORDER, weights))
    scale = target_total / base_sum
    raw = [weights[i] * scale for i in range(3)]
    floors = [int(x) for x in raw]
    rem = target_total - sum(floors)
    order = sorted(range(3), key=lambda i: (raw[i] - floors[i], i), reverse=True)
    for k in range(rem):
        floors[order[k]] += 1
    return list(zip(SLICE_ORDER, floors))


def _build_batch_plan(
    target_total: int, weights: tuple[int, int, int]
) -> list[tuple[str, int, str]]:
    """每切片条数全部由 Gemini 分批生成（每批最多 BATCH_CHUNK）。"""
    segments = _scaled_segment_totals(target_total, weights)
    plan: list[tuple[str, int, str]] = []
    for slice_id, total in segments:
        for size in _batch_sizes(total):
            plan.append(("gemini", size, slice_id))
    return plan


def _slice_totals(
    target_total: int, weights: tuple[int, int, int]
) -> dict[str, int]:
    return {sid: n for sid, n in _scaled_segment_totals(target_total, weights)}


def _split_train_test(
    rows: list[dict],
    test_n: int,
    *,
    segment_counts: dict[str, int],
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """按 slice 分层划分 test 集，各 slice 内 test 条数与 segment_counts 占总量比例一致（最大余数法）。"""
    if test_n < 0 or test_n > len(rows):
        raise ValueError("split-test 无效")
    total_n = len(rows)
    if total_n != sum(segment_counts.values()):
        raise ValueError("segment_counts 与 rows 条数不一致")
    rng = random.Random(seed)
    slices = SLICE_ORDER
    raw_test = [test_n * segment_counts[s] / total_n for s in slices]
    floors = [int(x) for x in raw_test]
    rem = test_n - sum(floors)
    order = sorted(range(3), key=lambda i: (raw_test[i] - floors[i], i), reverse=True)
    for k in range(rem):
        floors[order[k]] += 1
    test_per_slice = dict(zip(slices, floors))
    by_slice: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_slice[r["_meta"]["slice"]].append(r)
    train: list[dict] = []
    test: list[dict] = []
    for sid in slices:
        pool = by_slice[sid][:]
        rng.shuffle(pool)
        tn = test_per_slice[sid]
        if len(pool) < tn:
            raise ValueError(f"slice {sid}: 需要 {tn} 条测试集，实际仅 {len(pool)} 条")
        test.extend(pool[:tn])
        train.extend(pool[tn:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


SLICE_LABEL_ZH: dict[str, str] = {
    "private_domain": "私域语境（内部项目 ModelHub-Serve，基于 FastAPI+Transformers）",
    "dev_general_no_rag": "开发通识（无需查库）",
    "misleading_edge": "误导向/边界",
}

# 模型名改成读 src/config.py 的 data_gen_model（默认 gemini-3.8-flash），也可用 --model 覆盖
SLEEP_SECONDS = 4.0

# 用量统计：每批累加，最后估算花费（美元 / 百万 token，评测时固定单价；输出含思考 token）
GEMINI_PRICES = {
    "gemini-4-argon": (2.00, 10.00),          # 2026-11-30 前的价格，之后翻倍
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3.8-flash": (0.75, 3.75),
}
USAGE = {"input": 0, "output": 0, "thinking": 0}

# —— Phase 2A 路由输出 schema（嵌入各切片 prompt；不缩短工具列表，见 TOOL_LIST_AND_PARAMS_BLOCK）——
def _output_schema_block(*, bucket: str, kind_counts: dict[str, int], n: int) -> str:
    ct = kind_counts["tool_call"]
    cn = kind_counts["no_tool"]
    cc = kind_counts["clarify_needed"]
    return f"""
【必须遵守的输出 Schema（Phase 2A 单轮路由；每条对象字段一致；仅按 decision_type 填有效字段）】
本批共 {n} 条；请**严格**按下面条数分配 decision_type：tool_call **{ct}** 条、no_tool **{cn}** 条、clarify_needed **{cc}** 条（三类条数之和必须等于 {n}）。
bucket 本批固定为："{bucket}"

每条对象须包含以下键：
- "sample_id": 字符串，建议形如 "{bucket}_001" 递增，本批内唯一即可。
- "bucket": 固定 "{bucket}"
- "query": 用户问题（中文为主，可夹英文术语）
- "decision_type": 只能是 "tool_call" | "no_tool" | "clarify_needed"

当 decision_type == "tool_call":
- "tool_name": 上述六工具之一（与 TOOL_LIST 完全一致）
- "arguments": 与该工具 schema 一致的对象，字段完整、类型正确、可执行
- "clarify_answer": null
- "oracle_context": 字符串或 null；若为 search_codebase，可写极短检索意图说明（便于审计，非替代 arguments）

当 decision_type == "no_tool":
- "tool_name": null
- "arguments": null
- "clarify_answer": null
- "oracle_context": null
- 含义：本轮无需调用任何工具即可完成路由判断（例如与工具无关、或已明确无需工具）。**禁止**编造本仓库具体文件、默认配置、镜像名等未经验证的事实。

当 decision_type == "clarify_needed":
- "tool_name": null
- "arguments": null
- "clarify_answer": **1～2 句**中文，仅用于追问缺失信息或点明歧义；禁止教程式长文、禁止身份表演套话
- "oracle_context": null

返回**严格 JSON 数组**，顶层仅此数组，不要 markdown。示例（结构示意）：
[
  {{
    "sample_id": "{bucket}_001",
    "bucket": "{bucket}",
    "query": "...",
    "decision_type": "tool_call",
    "tool_name": "search_codebase",
    "arguments": {{"query": "...", "top_k": 6}},
    "clarify_answer": null,
    "oracle_context": null
  }},
  {{
    "sample_id": "{bucket}_002",
    "bucket": "{bucket}",
    "query": "...",
    "decision_type": "no_tool",
    "tool_name": null,
    "arguments": null,
    "clarify_answer": null,
    "oracle_context": null
  }},
  {{
    "sample_id": "{bucket}_003",
    "bucket": "{bucket}",
    "query": "...",
    "decision_type": "clarify_needed",
    "tool_name": null,
    "arguments": null,
    "clarify_answer": "请贴出完整 Traceback 或报错行号。",
    "oracle_context": null
  }}
]
"""


# 与 lora/generate_lora_data_fastapi_transformers.py 中【Agent 工具列表与参数】一节一字不差（便于 tool_call 监督对齐）
TOOL_LIST_AND_PARAMS_BLOCK = """
【Agent 工具列表与参数（务必遵守工具名与字段）】
1. 工具名: `analyse_code`
   - 触发场景: 静态代码问题分析。
   - 参数: {{"code": "<必填，一段存在潜在问题的 Python 源代码>"}}
2. 工具名: `search_codebase`
   - 触发场景: 询问某个模块、逻辑在哪，或查询实现/配置。
   - 参数: {{"query": "<必填，检索关键词/短句>", "top_k": <选填整数，默认6>}}
3. 工具名: `analyse_log`
   - 触发场景: 报错信息、异常堆栈分析。
   - 参数: {{"log": "<必填，一段包含 traceback 的报错文本>"}}
4. 工具名: `generate_dockerfile`
   - 触发场景: 容器化、Docker、部署依赖描述。
   - 参数: {{"project_description": "<必填，项目运行环境或依赖描述>"}}
5. 工具名: `generate_tests`
   - 触发场景: 单元测试（pytest）或边界覆盖。
   - 参数: {{"code": "<必填，需要被测试的函数代码>", "description": "<选填，额外说明>"}}
6. 工具名: `complete_code`
   - 触发场景: 续写残缺代码。
   - 参数: {{"context": "<必填，用户已有的残缺代码>"}}
"""


DATA_GENERATOR_PROMPT_PRIVATE_DOMAIN = (
    """你是一个 AI 数据集构建者。正在为代码 Agent 生成 Function Calling 训练数据。
用户是内部项目 ModelHub-Serve 的开发者（见下方背景，项目基于 FastAPI + Transformers），问题应围绕这个项目的目录、类名、概念与典型工程任务，语气贴近真实开发者。

为便于一次读全，请按顺序使用三块信息：① 内部项目背景（供你取材）② 工具名与参数（当 decision_type 为 tool_call 时必须遵守的契约）③ 文末 Phase 2A 路由 Schema（decision_type、tool_name、arguments、clarify_answer 等字段与 JSON 示例）。

【内部项目背景】
{project_context}

（上表为语料：写 query 时可自然引用其中的路径、类名、术语或典型工程问题，不必复述长段背景。）
"""
    + TOOL_LIST_AND_PARAMS_BLOCK
    + """

【Phase 2A 单轮工具路由说明（必须与文末 Schema 一致）】
本批数据用于「单轮工具路由对齐」：监督信号是**路由决策本身**，不是长篇自然语言回答。
- 每条样本必须作出且仅作出路由决策：decision_type 只能是 tool_call、no_tool、clarify_needed 之一。
- decision_type == tool_call：必须给出可被执行的 tool_name 与 arguments；参数须与上列六工具 schema 完全一致，信息密度高，避免空洞占位。
- decision_type == no_tool：表示当前单轮判断下无需调用任何工具。**不得**编造「本仓库」「我们项目」的具体实现、路径、默认镜像或配置参数。
  本切片的 no_tool 优先写成：**提到本项目或它用的框架，但问的是 FastAPI / Transformers 的通用用法**（见背景最后一节）。
- decision_type == clarify_needed：信息不足或存在歧义、必须先澄清再安全选工具。clarify_answer 仅 1～2 句中文，用于追问或点明歧义；禁止教程式展开、禁止堆砌身份表演。
- 不要输出 response_kind；不要输出 direct_answer 字段；澄清仅使用 clarify_answer。

字段与各类条数分配见下方 Schema。

{schema_block}
"""
)


DATA_GENERATOR_PROMPT_DEV_GENERAL = (
    """你是一个 AI 数据集构建者。正在为代码 Agent 生成 Function Calling 训练数据。

【本批数据切片：开发通识 · 无需查库】
user_query 应面向**软件开发常见话题**（语言与运行时、调试思路、依赖与包管理、Git、容器与部署概念、测试与 CI 思路、通用 HTTP/API 用法等），且**不需要**依赖「某个具体业务仓库内的私有目录结构或内部文件名」才能理解或处理。
优先生成可通过常识、公开文档级知识、或题干中已给出的代码/日志片段本身就能处理的请求；避免把「必须检索某代码库才能知道放哪」作为唯一前提。

tool_call 时只能从下列工具里选**除 search_codebase 以外**的 5 个（处理题干里已给出的代码、日志或需求）。
**本切片禁止 search_codebase**：通识问题不需要查任何私有代码库，和 Agent 运行时的规则一致（[time redacted] 修订；旧版允许“泛化检索”，与运行时规则、测试金标矛盾）。

【必须包含的一类题：提到常用框架，但问的是通用用法】
本批 no_tool 中**至少三分之一**要点名 FastAPI、Pydantic、Uvicorn、Transformers、PEFT 等常用库的具体 API 或概念，但问的是**通用写法/原理**，例如「FastAPI 的路由怎么注册？」「Depends 和直接调用函数有什么区别？」「AutoTokenizer 的 padding_side 是干嘛的？」。
这类题看起来和某个用这些框架的项目很近，但不需要看任何私有代码，应判为 no_tool。

"""
    + TOOL_LIST_AND_PARAMS_BLOCK
    + """

【Phase 2A 单轮工具路由说明（必须与文末 Schema 一致）】
本切片侧重「开发通识、无需强依赖私库路径」的 query 分布：no_tool 与 tool_call 均常见；当信息不足以安全选工具时，应使用 clarify_needed。
- decision_type、tool_name、arguments、clarify_answer 的约束同私域切片；**禁止**在 no_tool 或 clarify_answer 中捏造具体仓库事实。
- 不要输出 response_kind 或 direct_answer 字段。

字段与各类条数分配见下方 Schema。

{schema_block}
"""
)


DATA_GENERATOR_PROMPT_MISLEADING_EDGE = (
    """你是一个 AI 数据集构建者。正在为代码 Agent 生成 Function Calling 训练数据。

【本批数据切片：误导向 / 边界 / 压力】
user_query 应包含**歧义指令、缺失关键信息、表面矛盾的需求、易误判工具的情形、或边界条件**（例如：日志与代码片段指向不同工具、同一句话同时像两种工具、超出工具能力边界等），用于训练模型**谨慎选择工具**与**澄清**能力。
每条样本在 tool_call 时须有合理的 tool_name 与 arguments；可刻意制造「需要推理才能选对工具」的情形，但 arguments 必须自洽、可解析。

"""
    + TOOL_LIST_AND_PARAMS_BLOCK
    + """

【Phase 2A 单轮工具路由说明（必须与文末 Schema 一致）】
本切片侧重误导向、缺信息、易误判工具的情形：应提高 clarify_needed 的合理比例；tool_call 仅在「选工具确实合理」时出现，arguments 必须自洽。
- 本切片的 no_tool 优先写成**“看起来像在问某个项目，其实是通用原理或通用写法”**，例如「我们这种项目一般为什么用 FastAPI？」「如果要在 FastAPI 里加一个新路由，通常怎么写？」：带着“我们项目 / 这种项目”的字眼，但不需要看任何私有代码。
- 同上：decision_type 三分类、clarify_answer 短句、禁止捏造仓库细节、禁止 response_kind / direct_answer 字段。

字段与各类条数分配见下方 Schema。

{schema_block}
"""
)


def _build_slice_prompt(slice_id : str,n : int,mix : dict | None = None)->str:
    bucket = SLICE_TO_BUCKET[slice_id]
    mix = MIX_BY_SLICE[slice_id] if mix is None else mix
    kind_counts = _allocate_kind_counts(n, mix)
    schema_block = _output_schema_block(bucket=bucket, kind_counts=kind_counts, n=n)

    if slice_id == "private_domain":
        return DATA_GENERATOR_PROMPT_PRIVATE_DOMAIN.format(
            project_context=PROJECT_CONTEXT_INTERNAL,
            schema_block=schema_block,
        )
    if slice_id == "dev_general_no_rag":
        return DATA_GENERATOR_PROMPT_DEV_GENERAL.format(schema_block=schema_block)
    if slice_id == "misleading_edge":
        return DATA_GENERATOR_PROMPT_MISLEADING_EDGE.format(schema_block=schema_block)
    raise ValueError(f"unknown slice: {slice_id!r}")


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```json"):
        t = t[7:]
    elif t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def _parse_json_array(raw: str) -> list:
    data = json.loads(_strip_json_fence(raw))
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
    return data if isinstance(data, list) else []


# 补题（--supplement）：只生成私域切片、派单类型固定，每批再追加一段额外要求
# 起因：路由模型学成了“题里有‘我们的’或文件路径才查库”，RepoAgent 里不带这些记号的问题就漏查
SUPPLEMENTS: dict[str, tuple[dict[str, float], str]] = {
    "implicit_search": (
        {"tool_call": 1.0, "no_tool": 0.0, "clarify_needed": 0.0},
        """
【本批额外要求：隐式查库题（优先级高于上面的语气场景）】
- 全部是 tool_call，tool_name 一律为 search_codebase。
- query 直接点名 ModelHub-Serve 的模块、类、函数或机制，问的是这个项目怎么实现、在哪、怎么配置，比如「BatchScheduler 的超时是怎么处理的」「限流阈值默认多少」。
- query 里**禁止**出现“我们”“我们的”“咱们”“本项目”“这个项目”“项目里”等字样，也**禁止**出现文件路径（不要出现 / 或 .py）。
- arguments.query 写成适合检索的关键词（可以带类名、函数名）。""",
    ),
    "concept_no_tool": (
        {"tool_call": 0.0, "no_tool": 1.0, "clarify_needed": 0.0},
        """
【本批额外要求：术语像本项目，其实是通用概念（优先级高于上面的语气场景）】
- 全部是 no_tool。
- query 用和 ModelHub-Serve 背景相近的术语（动态批处理、模型预热、显存回收、LoRA 微调、限流、JWT 鉴权、灰度发布、tokenizer 缓存等），但问的是**通用原理或概念**，比如「动态批处理的原理是什么」，不看任何私有代码也能回答。
- query 里**不要**点名 ModelHub-Serve 自己的类名、函数名或文件名，也不要出现“我们”“本项目”等字样。""",
    ),
}
ID_TAG = ""  # 补题时 sample_id 加上类型，比如 domain_implicit_0001，避免和已有数据重号

_bucket_seq: dict[str, int] = {b: 0 for b in SLICE_TO_BUCKET.values()}


def _finalize_row(row: dict, *, slice_id: str) -> dict:
    """补全 bucket、稳定 sample_id、兼容旧字段名。"""
    bucket = SLICE_TO_BUCKET[slice_id]
    row["bucket"] = bucket
    if row.get("query") is None and row.get("user_query") is not None:
        row["query"] = row["user_query"]
    _bucket_seq[bucket] = _bucket_seq.get(bucket, 0) + 1
    row["sample_id"] = f"{bucket}_{ID_TAG}{_bucket_seq[bucket]:04d}"
    row.setdefault("sft_phase", "2a_routing")
    return row


def _generate_openai_compatible(
    prompt: str, *, model: str, batch_size: int, max_retries: int = 6
) -> list:
    """走 OpenAI 兼容接口（中转站）生成；配置见 data_gen_base_url / data_gen_api_key。"""
    from openai import OpenAI
    from src.config import settings

    client = OpenAI(api_key=settings.data_gen_api_key, base_url=settings.data_gen_base_url, timeout=600)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=1.0,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"请生成本批共 {batch_size} 条，严格按 system 中的 Phase 2A 路由 Schema 与 decision_type 条数分配，只输出 JSON 数组。"},
                ],
            )
            if response.usage is not None:
                USAGE["input"] += response.usage.prompt_tokens or 0
                USAGE["output"] += response.usage.completion_tokens or 0
            return _parse_json_array(response.choices[0].message.content or "")
        except Exception as e:
            print(f"  [OpenAI 兼容接口] 第 {attempt + 1} 次失败: {e}")
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 15)
    return []


def _generate_gemini(
    prompt: str, *, model: str, batch_size: int, max_retries: int = 6
) -> list:
    # 配了中转站就走 OpenAI 兼容接口，否则直连 Gemini
    from src.config import settings
    if settings.data_gen_base_url.strip():
        return _generate_openai_compatible(prompt, model=model, batch_size=batch_size, max_retries=max_retries)

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError("请 pip install google-genai 并在 .env 配置 google_api_key") from e

    from src.config import settings

    api_key = settings.google_api_key.strip()
    if not api_key:
        raise RuntimeError("请在 .env 中配置 google_api_key（见 src/config.py）")

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=prompt,
        response_mime_type="application/json",
        temperature=1.0,
    )
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                config=config,
                contents=[
                    f"请生成本批共 {batch_size} 条，严格按 system 中的 Phase 2A 路由 Schema 与 decision_type 条数分配，只输出 JSON 数组。"
                ],
            )
            # 记下 token 用量，最后估算花费
            um = response.usage_metadata
            if um is not None:
                USAGE["input"] += um.prompt_token_count or 0
                USAGE["output"] += um.candidates_token_count or 0
                USAGE["thinking"] += um.thoughts_token_count or 0
            return _parse_json_array(response.text)
        except Exception as e:
            print(f"  [Gemini] 第 {attempt + 1} 次失败: {e}")
            if attempt < max_retries - 1:
                # 免费档常报 503（太忙）和 SSL 断连，等待逐次加长：15、30、45… 秒
                time.sleep((attempt + 1) * 15)
    return []


def _parse_segment_ratio(s: str) -> tuple[int, int, int]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("segment-ratio 须为三个整数，逗号分隔，如 300,100,100 或 70,15,15")
    try:
        w = tuple(int(x) for x in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e
    if any(x < 0 for x in w) or sum(w) == 0:
        raise argparse.ArgumentTypeError("segment-ratio 须为非负整数且和大于 0")
    return w  # type: ignore[return-value]


def main()->None:
    #1.先检查所有输出，避免生成完才发现不能保存
    parser = argparse.ArgumentParser(description="Phase 2A 路由 JSONL 生成（Gemini）")
    parser.add_argument(
        "--target",
        type=int,
        default=TARGET_TOTAL,
        help=f"生成总条数（默认 {TARGET_TOTAL}）",
    )
    parser.add_argument(
        "--segment-ratio",
        type=_parse_segment_ratio,
        default=DEFAULT_SEGMENT_WEIGHTS,
        metavar="W,W,W",
        help="三切片权重 private:dev_general:misleading，如 300,100,100（默认）或 70,15,15",
    )
    parser.add_argument(
        "--split-test",
        type=int,
        default=0,
        metavar="N",
        help="生成完成后按切片分层划分 N 条为测试集，写出 _train / _test；0 表示不划分",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="分层划分测试集时的随机种子（默认 42）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印分批计划，不调 API")
    parser.add_argument("--model", default=None, help="Gemini 模型名，默认读 .env / src/config.py 的 data_gen_model")
    parser.add_argument("--supplement", choices=list(SUPPLEMENTS), default=None, help="补题：implicit_search（隐式查库）/ concept_no_tool（术语像项目的通用概念）")
    parser.add_argument("--out-dir",type=Path,help="输出目录，已有同名文件时拒绝覆盖")
    args = parser.parse_args()

    from src.config import settings
    gen_model = args.model or settings.data_gen_model
    # 固定随机种子：每批抽到哪种提问风格可以复现
    random.seed(args.split_seed)

    target_total = args.target
    weights: tuple[int, int, int] = args.segment_ratio
    extra_prompt = ""
    global ID_TAG
    ID_TAG = ""
    _bucket_seq.update(dict.fromkeys(_bucket_seq,0))
    USAGE.update(dict.fromkeys(USAGE,0))
    mix_by_slice = dict(MIX_BY_SLICE)
    if args.supplement:
        # 补题：只要私域切片，派单类型按补题要求固定
        weights = (1, 0, 0)
        mix_by_slice["private_domain"], extra_prompt = SUPPLEMENTS[args.supplement]
        ID_TAG = args.supplement.split("_")[0] + "_"
    split_test_n = args.split_test
    if split_test_n < 0 or split_test_n >= target_total:
        parser.error("--split-test 须满足 0 <= N < --target")

    batch_plan = _build_batch_plan(target_total, weights)
    expected_slice_totals = _slice_totals(target_total, weights)

    out_dir = args.out_dir or ROOT / "experiments" / "routing_data" / "outputs"
    out_path = out_dir / (f"ft_qa_{target_total}_{args.supplement}.jsonl" if args.supplement else f"ft_qa_{target_total}.jsonl")
    outputs = [out_path]
    if split_test_n:
        outputs += [out_path.with_stem(out_path.stem + suffix) for suffix in ("_train","_test")]
    if not args.dry_run:
        if out_dir.exists() and not out_dir.is_dir():
            parser.error(f"输出目录不是文件夹：{out_dir}")
        for path in outputs:
            if path.exists() or path.is_symlink():
                parser.error(f"拒绝覆盖已有文件，请用 --out-dir 指定新目录：{path}")

    total = sum(n for _, n, _ in batch_plan)
    print(f"TARGET_TOTAL={target_total} | segment_ratio={weights} | 计划批次数: {len(batch_plan)}，总条数: {total} | 模型: {gen_model}", flush=True)
    print("切片条数:", expected_slice_totals, flush=True)
    print("切片与 bucket: private_domain→domain, dev_general_no_rag→far_general, misleading_edge→misleading_edge", flush=True)
    print("各切片 MIX:", mix_by_slice, flush=True)
    if args.dry_run:
        for i, (prov, n, sid) in enumerate(batch_plan):
            label = SLICE_LABEL_ZH.get(sid, sid)
            b = SLICE_TO_BUCKET[sid]
            kc = _allocate_kind_counts(n, mix_by_slice[sid])
            print(
                f"  {i+1:02d} {prov} x {n} | {sid} → bucket={b} | kind_counts={kc}",
                flush=True,
            )
        if split_test_n:
            print(f"  [dry-run] 将划分测试集 split_test={split_test_n}，seed={args.split_seed}", flush=True)
        print("dry-run 结束。", flush=True)
        return

    #2.每批数量和字段都合格才收下，失败时不写正式数据
    all_rows: list[dict] = []
    for bi, (provider, n, slice_id) in enumerate(batch_plan):
        scenario = random.choice(SCENARIOS)
        base = _build_slice_prompt(slice_id,n,mix_by_slice[slice_id])
        prompt = base + (
            f"\n\n——————————————————————————————\n"
            f"【本批次语气与场景限制（仅影响 query 风格）】\n"
            f"以上 Phase 2A 路由规则与 Schema 仍然全部适用。下列场景用于统一**用户问题**的语气与人设（query 文本），"
            f"**不要**让 clarify_answer 变成身份表演或长篇客套；clarify_answer 仍须 1～2 句专业追问。\n"
            f"请基于以下场景生成这 {n} 条数据的 query，且 decision_type 条数与字段仍须符合 Schema：\n{scenario}"
        ) + extra_prompt  # 补题时追加额外要求，放在语气场景之后

        print(f"\n>>> 批次 {bi + 1}/{len(batch_plan)} | {provider} | n={n} | slice={slice_id}", flush=True)
        print(f"    {SLICE_LABEL_ZH.get(slice_id, slice_id)} | kind配比 {_allocate_kind_counts(n, mix_by_slice[slice_id])}", flush=True)

        batch = _generate_gemini(prompt, model=gen_model, batch_size=n)

        if not isinstance(batch,list) or len(batch) != n:
            raise RuntimeError(f"第 {bi + 1} 批生成失败或条数不符，需要 {n} 条；未写入数据")
        for row in batch:
            if not isinstance(row, dict):
                raise ValueError(f"第 {bi + 1} 批包含非对象记录；未写入数据")
            _finalize_row(row, slice_id=slice_id)
            row["_meta"] = {
                "provider": provider,
                "batch_index": bi + 1,
                "scenario": scenario,
                "slice": slice_id,
                "slice_label_zh": SLICE_LABEL_ZH[slice_id],
                "sft_phase": "2a_routing",
            }
            errors = [issue.message for issue in collect_issues(row) if issue.severity == "ERROR"]
            if errors:
                raise ValueError(f"{row['sample_id']}：{'；'.join(errors)}；未写入数据")
            all_rows.append(row)

        print(f"    本批得到 {len(batch)} 条，累计 {len(all_rows)}", flush=True)
        time.sleep(SLEEP_SECONDS)

    # 估算花费（思考 token 按输出价计）
    p_in, p_out = GEMINI_PRICES.get(gen_model, (0.0, 0.0))
    cost = (USAGE["input"] * p_in + (USAGE["output"] + USAGE["thinking"]) * p_out) / 1e6
    print(f"token 用量: 输入 {USAGE['input']} | 输出 {USAGE['output']} | 思考 {USAGE['thinking']} | 估算花费约 ${cost:.2f}", flush=True)

    #3.先完成分层，再创建输出；切分失败时也不留下全量正式文件
    datasets = [all_rows]
    if split_test_n:
        train_rows, test_rows = _split_train_test(
            all_rows,
            split_test_n,
            segment_counts=expected_slice_totals,
            seed=args.split_seed,
        )
        datasets.extend((train_rows,test_rows))
    #4.先独占创建全部目标；保存失败只清理本次新文件，不动已有数据
    out_dir.mkdir(parents=True,exist_ok=True)
    created = []
    try:
        with ExitStack() as stack:
            streams = []
            for path in outputs:
                streams.append(stack.enter_context(path.open("x",encoding="utf-8")))
                created.append(path)
            for stream,rows in zip(streams,datasets):
                for row in rows:
                    stream.write(json.dumps(row,ensure_ascii=False) + "\n")
    except BaseException:
        for path in created:
            path.unlink()
        raise
    for path,rows in zip(outputs,datasets):
        print(f"完成，共写入 {len(rows)} 行 -> {path}",flush=True)


if __name__ == "__main__":
    main()
