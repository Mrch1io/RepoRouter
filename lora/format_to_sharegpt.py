# 将 Json 转为微调用 ShareGPT 格式。
#
# 与 experiments/routing_data 的关系：
# - Phase 2A（`generate_data.py` → `ft_qa_*.jsonl`）：见 `ROUTING_TASK.md`。
#   每条含 `decision_type` ∈ tool_call | no_tool | clarify_needed；assistant 监督为**单行路由 JSON**
#   （decision_type + tool_name + arguments + clarify_answer + oracle_context），
#   system 使用 `PHASE_2A_ROUTING_SYSTEM_PROMPT`（与生成侧工具块一致，不缩短）。
# - 历史 archive/early-experiments/lora/repoagent_train_data.json：assistant 仅为工具调用 JSON（name + arguments）。
# - 旧混合 schema：见下方 `response_kind`（direct_answer / clarify 等）；混合数据可用 `MIXED_SYSTEM_PROMPT`。
# - archive/early-experiments/lora/eval_lora.py：旧版假定 gold 可 json.loads 且含 name；Phase 2A 样本需按 decision_type 评测。
#
# 逐条判定：若存在 `decision_type` 或 `sft_phase`/`_meta.sft_phase` 为 2a_routing → Phase 2A；
# 否则未设置 response_kind 时与旧版一致（tool_call）。

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
# 1. 真实业务 System Prompt (保持一字不差)
USER_RUNTIME_PROMPT = """你是一个拥有多智能体协作能力的研发助手。
你有多个的辅助工具（如静态代码分析、源码检索、日志分析等）。
请严格遵守以下指令：
1. 分析用户问题，如果需要查阅代码、分析 Bug 或生成配置，必须毫不犹豫地调用对应工具。
2. 除非明显缺少信息，否则不要重复调用工具。每个问题最多调用 1 次 search_codebase。
3. 必须用中文回答。"""

# 2. 模拟底层 SDK 拼接的“工具描述与格式要求”
# 这样不仅维持了你的人设，还给了模型明确的输出规范
TOOL_SCHEMA_INSTRUCTION = """
【可用工具集合】
1. `analyse_code`: 静态分析Python代码规范。参数: {"code": "<Python代码>"}
2. `search_codebase`: 检索相关代码。参数: {"query": "<检索关键词>", "top_k": <整数>}
3. `analyse_log`: 分析报错日志。参数: {"log": "<报错日志/Traceback>"}
4. `generate_dockerfile`: 生成Docker配置。参数: {"project_description": "<环境要求>"}
5. `generate_tests`: 生成单元测试。参数: {"code": "<目标代码>", "description": "<额外说明>"}
6. `complete_code`: 补全残缺代码。参数: {"context": "<残缺代码片段>"}

【工具调用格式】
如果你需要调用工具，请务必直接输出如下纯 JSON 格式（不要附带任何解释性文本或 markdown 标记）：
{"name": "工具名称", "arguments": {"参数名": "参数值"}}
"""

FINAL_SYSTEM_PROMPT = f"{USER_RUNTIME_PROMPT}\n{TOOL_SCHEMA_INSTRUCTION}"

# 与「纯工具」数据混训或单独训「可拒答/可直答」时使用；推理侧需与之一致。
MIXED_SYSTEM_PROMPT = f"""{USER_RUNTIME_PROMPT}
4. 若问题无需查阅代码、运行工具或生成配置即可简洁回答，请直接用中文作答，不要输出工具 JSON。
5. 若用户表述有歧义、缺关键信息或可能选错工具，可先简短澄清或说明取舍，再决定是否调用工具；必要时再输出工具 JSON。
{TOOL_SCHEMA_INSTRUCTION}"""

# 与 experiments/routing_data/generate_data.py 中 TOOL_LIST_AND_PARAMS_BLOCK 一字不差
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

PHASE_2A_ROUTING_SYSTEM_PROMPT = f"""{USER_RUNTIME_PROMPT}
【Phase 2A 单轮工具路由】
本阶段监督目标是**单轮路由决策**（结构化 JSON），不是长篇自然语言解答。请分析用户问题后，只输出**一行**可解析的 JSON 对象，不要输出 markdown 围栏、不要前后解释或自然语言段落。
- decision_type 只能是 "tool_call" | "no_tool" | "clarify_needed"。
- tool_call：必须给出与下列列表一致的 tool_name 与可执行 arguments；clarify_answer 为 null；oracle_context 可为 null，或在 search_codebase 时给极短检索意图说明（非替代 arguments）。
- no_tool：tool_name、arguments、clarify_answer、oracle_context 均为 null；表示本轮无需调用任何工具；**禁止**编造本仓库具体路径、默认配置、镜像名等未验证事实。
- clarify_needed：tool_name、arguments、oracle_context 为 null；clarify_answer 为 1～2 句中文，仅追问缺失信息或点明歧义；禁止教程式长文与身份表演套话。
- 禁止输出 response_kind、direct_answer 字段。
{TOOL_LIST_AND_PARAMS_BLOCK}
【路由输出 JSON 形状（单行、无换行）】
{{"decision_type":"tool_call","tool_name":"search_codebase","arguments":{{"query":"关键词","top_k":6}},"clarify_answer":null,"oracle_context":null}}
{{"decision_type":"no_tool","tool_name":null,"arguments":null,"clarify_answer":null,"oracle_context":null}}
{{"decision_type":"clarify_needed","tool_name":null,"arguments":null,"clarify_answer":"请补充完整 Traceback 或报错行号。","oracle_context":null}}
"""

# 路由模型要知道“当前仓库里有什么”，才能判断一个问题是不是在问本仓库。
# 做法：system prompt 开头加一段【当前仓库】简介。
#   训练：用 ModelHub-Serve（生成训练数据时用的虚构内部项目，从 PROJECT_CONTEXT_INTERNAL 压缩来）
#   历史评测：保留 87 题完整提示词快照；运行时：读取当前检索索引的简介
# 两份格式一致：第一行“项目：名字（一句话定位）”，后面每行“文件：类名、函数名”
MODELHUB_REPO_SUMMARY = """项目：ModelHub-Serve（公司内部的模型推理与微调平台，基于 FastAPI + Transformers）
app/main.py：create_app
app/api/routers/inference.py：/v1/generate、/v1/embed
app/api/routers/jobs.py：/v1/finetune、/v1/jobs/{job_id}
app/api/deps.py：get_current_user、get_registry
app/core/config.py：Settings
app/services/model_registry.py：ModelRegistry、ModelRegistry.get、ModelRegistry.evict
app/services/batch_scheduler.py：BatchScheduler、BatchScheduler.submit、BatchScheduler._flush
app/services/tokenizer_cache.py：TokenizerCache
app/workers/finetune_worker.py：FinetuneJob、FinetuneJob.run
app/middleware/rate_limit.py：限流中间件
app/middleware/auth.py：JWT 鉴权
app/schemas/：GenerateRequest、GenerateResponse、FinetuneSpec"""


#用 ast 扫 src/ 下每个 .py，按“文件：类名、方法名、函数名”一行一个文件（跳过 __init__ 这类魔法方法和没定义东西的文件）
def build_repo_summary(root, *, name: str, desc: str, src: str = "src", max_names: int = 12) -> str:
    import ast
    root = Path(root)
    lines = [f"项目：{name}（{desc}）"]
    for p in sorted((root / src).rglob("*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        names = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                names.append(node.name)
                names += [f"{node.name}.{f.name}" for f in node.body
                          if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and not f.name.startswith("__")]
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.append(node.name)
        if names:
            #名字太多只留前 max_names 个（按定义顺序），控制 prompt 长度
            lines.append(f"{p.relative_to(root).as_posix()}：{'、'.join(names[:max_names])}")
    return "\n".join(lines)


def load_prompt_snapshot() -> str:
    """读取历史 87 题共用的完整提示词，不重新拼接当前模板或扫描源码。"""
    return Path(__file__).with_name("routing_prompt_snapshot.txt").read_text(encoding="utf-8")


#在 PHASE_2A_ROUTING_SYSTEM_PROMPT 开头放【当前仓库】；repo_summary 为空时就是旧版（无简介）prompt
def routing_system_prompt(repo_summary: str | None) -> str:
    if not repo_summary:
        return PHASE_2A_ROUTING_SYSTEM_PROMPT
    return f"【当前仓库】\n以下是当前私有代码仓库的文件与类名、函数名（自动整理，可能不全）：\n{repo_summary}\n\n{PHASE_2A_ROUTING_SYSTEM_PROMPT}"


#训练数据默认带 ModelHub 简介；--no-repo-summary 可以导出旧版
PHASE_2A_SYSTEM = routing_system_prompt(MODELHUB_REPO_SUMMARY)


def _is_phase_2a_row(item: dict) -> bool:
    if item.get("decision_type") is not None:
        return True
    if item.get("sft_phase") == "2a_routing":
        return True
    meta = item.get("_meta") or {}
    if meta.get("sft_phase") == "2a_routing":
        return True
    return False


def _assistant_routing_json(item: dict) -> str:
    """Phase 2A：与 ft_qa_*.jsonl 路由 schema 一致的单行 JSON。"""
    dt = item["decision_type"]
    if dt == "tool_call":
        out = {
            "decision_type": "tool_call",
            "tool_name": item.get("tool_name"),
            "arguments": item.get("arguments") if item.get("arguments") is not None else {},
            "clarify_answer": None,
            "oracle_context": item.get("oracle_context"),
        }
    elif dt == "no_tool":
        out = {
            "decision_type": "no_tool",
            "tool_name": None,
            "arguments": None,
            "clarify_answer": None,
            "oracle_context": None,
        }
    elif dt == "clarify_needed":
        out = {
            "decision_type": "clarify_needed",
            "tool_name": None,
            "arguments": None,
            "clarify_answer": item.get("clarify_answer") or "",
            "oracle_context": None,
        }
    else:
        raise ValueError(f"未知 decision_type: {dt!r}")
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def _assistant_turn_value(item: dict) -> str:
    """优先 Phase 2A 路由 JSON；否则按 response_kind（旧混合 schema / 纯工具）。"""
    if _is_phase_2a_row(item) and item.get("decision_type") is not None:
        return _assistant_routing_json(item)
    kind = item.get("response_kind", "tool_call")
    if kind == "tool_call":
        tool_call_dict = {"name": item["tool_name"], "arguments": item["arguments"]}
        return json.dumps(tool_call_dict, ensure_ascii=False)
    if kind == "direct_answer":
        return item["direct_answer"]
    if kind == "clarify":
        # 与 ft_qa_500 混合 schema 一致：澄清文本放在 direct_answer 字段
        return item.get("clarify_answer") or item["direct_answer"]
    raise ValueError(f"未知 response_kind: {kind!r}")


def _user_turn_value(item: dict) -> str:
    return item.get("query") or item.get("user_query", "")


def _system_prompt_for_item(item: dict) -> str:
    """Phase 2A 行 → 路由 system；否则 FINAL 或 MIXED。"""
    if _is_phase_2a_row(item):
        return PHASE_2A_SYSTEM
    if item.get("use_mixed_system"):
        return MIXED_SYSTEM_PROMPT
    return FINAL_SYSTEM_PROMPT


def convert_to_sharegpt(raw_data, *, mixed_system: bool = False, force_phase_2a_system: bool = False):
    """
    mixed_system=True 时忽略逐条 use_mixed_system，全部使用 MIXED_SYSTEM_PROMPT（适合整份都是混合切片）。
    force_phase_2a_system=True 时**每条**使用 PHASE_2A_ROUTING_SYSTEM_PROMPT（整份均为 Phase 2A JSONL 时）。
    """
    formatted_data = []
    for item in raw_data:
        assistant_value = _assistant_turn_value(item)
        if force_phase_2a_system:
            system_value = PHASE_2A_SYSTEM
        elif mixed_system:
            system_value = MIXED_SYSTEM_PROMPT
        else:
            system_value = _system_prompt_for_item(item)

        formatted_case = {
            "conversations": [
                {"from": "system", "value": system_value},
                {"from": "user", "value": _user_turn_value(item)},
                {"from": "assistant", "value": assistant_value},
            ]
        }
        formatted_data.append(formatted_case)

    return formatted_data


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}: line {line_no}: invalid JSON: {e}") from e
    return rows


def _main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert RepoRouter JSON / JSONL to ShareGPT (conversations + from/value)."
    )
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input .json (list) or .jsonl (one object per line).",
    )
    ap.add_argument(
        "--out-train",
        type=Path,
        required=True,
        help="Write training ShareGPT JSON array here.",
    )
    ap.add_argument(
        "--out-eval",
        type=Path,
        help="Optional: write eval ShareGPT JSON array (use with --eval-input or --split).",
    )
    ap.add_argument(
        "--eval-input",
        type=Path,
        help="Optional: separate eval JSONL; skips --split on main input.",
    )
    ap.add_argument(
        "--split",
        type=float,
        default=None,
        metavar="RATIO",
        help="If set (0,1), shuffle train input and hold out this fraction for --out-eval.",
    )
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed when using --split.")
    ap.add_argument(
        "--phase2a",
        action="store_true",
        help="Force PHASE_2A_ROUTING_SYSTEM_PROMPT for every row (ft_qa Phase 2A JSONL).",
    )
    ap.add_argument(
        "--mixed-system",
        action="store_true",
        help="Use MIXED_SYSTEM_PROMPT for every row (legacy mixed slices).",
    )
    ap.add_argument("--no-repo-summary", action="store_true", help="导出旧版 prompt（开头没有【当前仓库】简介）")
    args = ap.parse_args()
    #1.先检查参数和路径，出错时不动任何输出文件
    has_eval = args.eval_input is not None or args.split is not None
    if has_eval and args.out_eval is None:
        ap.error("eval data present but --out-eval not set")
    if args.out_eval is not None and not has_eval:
        ap.error("--out-eval requires --eval-input or --split")
    if args.split is not None:
        if not (0 < args.split < 1):
            ap.error("--split must be in (0, 1)")
        if args.eval_input is not None:
            ap.error("use either --eval-input or --split, not both")
    outputs = [args.out_train] + ([args.out_eval] if has_eval else [])
    inputs = {p.resolve() for p in (args.input, args.eval_input) if p is not None}
    if len({p.resolve() for p in outputs}) != len(outputs):
        ap.error("训练集和评测集不能写入同一路径")
    for path in outputs:
        if path.resolve() in inputs:
            ap.error(f"输出文件不能覆盖输入文件：{path}")
        if path.exists() or path.is_symlink():
            ap.error(f"拒绝覆盖已有文件，请指定新路径：{path}")
    if args.no_repo_summary:
        global PHASE_2A_SYSTEM
        PHASE_2A_SYSTEM = PHASE_2A_ROUTING_SYSTEM_PROMPT

    suffix = args.input.suffix.lower()
    if suffix == ".jsonl":
        raw_data = load_jsonl(args.input)
    else:
        with args.input.open("r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, list):
            raise SystemExit("--input JSON must be a list of objects")

    eval_rows: list[dict] | None = None
    if args.eval_input is not None:
        if args.eval_input.suffix.lower() == ".jsonl":
            eval_rows = load_jsonl(args.eval_input)
        else:
            eval_rows = json.loads(args.eval_input.read_text(encoding="utf-8"))
        if not isinstance(eval_rows, list):
            raise SystemExit("--eval-input JSON must be a list of objects")

    if args.split is not None:
        random.seed(args.seed)
        shuffled = list(raw_data)
        random.shuffle(shuffled)
        n_eval = max(1, int(len(shuffled) * args.split))
        eval_rows = shuffled[-n_eval:]
        raw_data = shuffled[:-n_eval]

    out_train = args.out_train
    out_eval = args.out_eval

    #2.两份数据都转换成功后再写，避免只写了一半才发现输入有错
    train_data = convert_to_sharegpt(
        raw_data,
        mixed_system=args.mixed_system,
        force_phase_2a_system=args.phase2a,
    )
    if eval_rows is not None:
        test_data = convert_to_sharegpt(
            eval_rows,
            mixed_system=args.mixed_system,
            force_phase_2a_system=args.phase2a,
        )
    #3.x模式只创建新文件，检查后即使有人抢先创建，也不会覆盖它
    out_train.parent.mkdir(parents=True, exist_ok=True)
    with out_train.open("x", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    print(f"训练集: {len(train_data)} 条 -> {out_train}")

    if eval_rows is not None:
        out_eval.parent.mkdir(parents=True, exist_ok=True)
        with out_eval.open("x", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        print(f"验证/测试集: {len(test_data)} 条 -> {out_eval}")


if __name__ == "__main__":
    _main()
