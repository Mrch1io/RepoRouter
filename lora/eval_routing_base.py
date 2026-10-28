"""
Phase 2A 路由测评：基座模型（单文件，可拷到 AutoDL 单独跑）。

默认路径与旧 eval_base.py 一致风格；用 --base-model / --test-data 覆盖。
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
import sys

# 参数合法率要用 src/agent/tools/schemas.py 里的工具清单。
# 整个项目在时能导入；只拷这一个文件走时，这一项显示 N/A
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from src.agent.tools.schemas import check_tool_args
except ImportError:
    check_tool_args = None

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

DEFAULT_BASE_MODEL = "/root/autodl-tmp/models/Qwen/Qwen3.5-9B"   # 基座换成 Qwen3.5（4B 用 --base-model 覆盖）
DEFAULT_TEST_DATA = "/root/LLaMA-Factory/data/eval_in_domain.json"


def _resolve_model_source(path_or_id: str) -> str:
    """本地路径转绝对路径；HuggingFace id 原样返回。"""
    p = Path(path_or_id).expanduser()
    if p.exists():
        return str(p.resolve())
    resolved = p.resolve()
    if resolved.exists():
        return str(resolved)
    return path_or_id


def _load_causal_lm(path_or_id: str) -> PreTrainedModel:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

    kw = dict(device_map="auto", trust_remote_code=True)
    # Qwen3.5 是多模态模型（Qwen3_5ForConditionalGeneration），LLaMA-Factory 也用这个类训练，
    # LoRA 权重名带 model.language_model.*；用纯文本类加载会对不上名字，LoRA 被悄悄跳过
    archs = AutoConfig.from_pretrained(path_or_id, trust_remote_code=True).architectures or []
    cls = AutoModelForImageTextToText if any("ConditionalGeneration" in a for a in archs) else AutoModelForCausalLM
    try:
        return cls.from_pretrained(path_or_id, dtype=torch.bfloat16, **kw)
    except TypeError:
        return cls.from_pretrained(path_or_id, torch_dtype=torch.bfloat16, **kw)

ALLOWED_TOOLS = frozenset(
    {
        "analyse_code",
        "complete_code",
        "generate_tests",
        "generate_dockerfile",
        "analyse_log",
        "search_codebase",
    }
)
ALLOWED_DECISIONS = frozenset({"tool_call", "no_tool", "clarify_needed"})
_FENCE = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)


def _strip_markdown_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = _FENCE.sub("", t)
        if t.endswith("```"):
            t = t[:-3].strip()
    return t


def parse_routing_json(text: str) -> dict[str, Any] | None:
    if not text or not text.strip():
        return None
    candidates = [text.strip(), _strip_markdown_fence(text)]
    decoder = json.JSONDecoder()
    for chunk in candidates:
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict) and "decision_type" in obj:
                return obj
        except json.JSONDecodeError:
            pass
        i = 0
        while i < len(chunk):
            j = chunk.find("{", i)
            if j < 0:
                break
            try:
                obj, _end = decoder.raw_decode(chunk, j)
                if (
                    isinstance(obj, dict)
                    and isinstance(obj.get("decision_type"), str)
                    and obj["decision_type"] in ALLOWED_DECISIONS
                ):
                    return obj
            except json.JSONDecodeError:
                pass
            i = j + 1
    return None


def is_valid_routing(obj: dict[str, Any]) -> bool:
    dt = obj.get("decision_type")
    if not isinstance(dt, str) or dt not in ALLOWED_DECISIONS:
        return False
    if dt == "tool_call":
        tn = obj.get("tool_name")
        return isinstance(tn, str) and tn in ALLOWED_TOOLS
    if dt == "no_tool":
        return obj.get("tool_name") is None and obj.get("arguments") is None
    if dt == "clarify_needed":
        ca = obj.get("clarify_answer")
        return isinstance(ca, str) and bool(ca.strip())
    return False


def read_gold_answer(case: dict[str, Any]) -> dict[str, Any] | None:
    try:
        gold_str = case["conversations"][2]["value"]
        g = json.loads(gold_str)
        if not isinstance(g, dict) or g.get("decision_type") not in ALLOWED_DECISIONS:
            return None
        return g
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None


def _norm_arguments(a: Any) -> Any:
    if a is None:
        return None
    if isinstance(a, dict):
        return json.dumps(a, sort_keys=True, ensure_ascii=False)
    return a


def arguments_equal(gold: dict[str, Any], pred: dict[str, Any]) -> bool:
    return _norm_arguments(gold.get("arguments")) == _norm_arguments(pred.get("arguments"))


@dataclass
class Phase2aMetrics:
    n_file_rows: int = 0
    skipped_bad_gold: int = 0
    n_evaluated: int = 0
    n_parse: int = 0
    n_parse_valid_shape: int = 0
    n_decision_match: int = 0
    n_gold_tool_call: int = 0
    n_tool_match: int = 0
    n_args_match: int = 0
    n_hallucination_tool: int = 0
    # 借鉴 BFCL：“该调时调了”（relevance）和“不该调时没调”（irrelevance）分开算
    n_relevance_ok: int = 0
    n_gold_no_call: int = 0
    n_irrelevance_ok: int = 0
    n_args_valid: int = 0

    def update(self, gold: dict[str, Any], pred: dict[str, Any] | None) -> None:
        self.n_evaluated += 1
        gold_call = gold.get("decision_type") == "tool_call"
        # 子集分母只由金标决定；解析失败留在分母中，按错误计。
        if gold_call:
            self.n_gold_tool_call += 1
        else:
            self.n_gold_no_call += 1
        if pred is None:
            return
        self.n_parse += 1
        pred_call = pred.get("decision_type") == "tool_call"
        if pred_call:
            tn = pred.get("tool_name")
            if not isinstance(tn, str) or tn not in ALLOWED_TOOLS:
                self.n_hallucination_tool += 1
        if not is_valid_routing(pred):
            return
        self.n_parse_valid_shape += 1
        if pred.get("decision_type") == gold.get("decision_type"):
            self.n_decision_match += 1
        if not gold_call:
            self.n_irrelevance_ok += not pred_call
        else:
            self.n_relevance_ok += pred_call
            if (
                pred.get("decision_type") == "tool_call"
                and pred.get("tool_name") == gold.get("tool_name")
            ):
                self.n_tool_match += 1
                if check_tool_args and check_tool_args(pred.get("tool_name"), pred.get("arguments")):
                    self.n_args_valid += 1
                if arguments_equal(gold, pred):
                    self.n_args_match += 1

    def report_lines(self) -> list[str]:
        g = max(self.n_evaluated, 1)
        gc = max(self.n_gold_tool_call, 1)
        return [
            "====== Phase 2A 路由测评 ======",
            f"样本总数（文件条数）: {self.n_file_rows}",
            f"金标可解析条数: {self.n_evaluated}（跳过金标损坏: {self.skipped_bad_gold}）",
            f"模型输出可解析为 JSON: {self.n_parse} ({100 * self.n_parse / g:.3f}%)",
            f"解析且角色字段合法: {self.n_parse_valid_shape} ({100 * self.n_parse_valid_shape / g:.3f}%)",
            f"decision_type 一致: {self.n_decision_match} ({100 * self.n_decision_match / g:.3f}%)",
            f"金标为 tool_call 的条数: {self.n_gold_tool_call}",
            f"tool_call 子集·工具名一致: {self.n_tool_match} ({100 * self.n_tool_match / gc:.3f}% of gold tool_call)",
            f"tool_call 子集·工具+arguments 一致: {self.n_args_match} ({100 * self.n_args_match / gc:.3f}% of gold tool_call)",
            f"预测 tool_call 但工具名非法/空: {self.n_hallucination_tool}（越低越好）",
            "—— BFCL 口径 ——",
            f"相关检测（该调工具时调了）: {self.n_relevance_ok}/{self.n_gold_tool_call} ({100 * self.n_relevance_ok / gc:.1f}%)",
            f"无关检测（不该调时没调）: {self.n_irrelevance_ok}/{self.n_gold_no_call} ({100 * self.n_irrelevance_ok / max(self.n_gold_no_call, 1):.1f}%)",
            (f"参数合法率（工具名对的里面，必填参数齐、类型对、没编参数）: {self.n_args_valid}/{self.n_tool_match} "
             f"({100 * self.n_args_valid / max(self.n_tool_match, 1):.1f}%)") if check_tool_args else "参数合法率: N/A（没找到 schemas.py）",
            "=" * 50,
        ]


def generate_reply(
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    case: dict[str, Any],
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> str:
    import torch

    message = [
        {"role": "system", "content": case["conversations"][0]["value"]},
        {"role": "user", "content": case["conversations"][1]["value"]},
    ]
    text = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,  # Qwen3.5 模板默认会先“思考”；路由只要一行 JSON，关掉（旧模板会忽略这个参数）
    )
    model_input = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
    output_ids = [
        out[len(inp) :] for inp, out in zip(model_input.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2A routing eval — base model")
    ap.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    ap.add_argument(
        "--test-data",
        type=Path,
        default=Path(DEFAULT_TEST_DATA),
        help="ShareGPT JSON 数组",
    )
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.1)
    args = ap.parse_args()

    from tqdm import tqdm
    from transformers import AutoTokenizer

    base_model = _resolve_model_source(args.base_model)
    print("正在加载分词器与基座模型…")
    print(f"  base_model = {base_model}\n")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = _load_causal_lm(base_model)
    model.eval()

    with args.test_data.open("r", encoding="utf-8") as f:
        test_cases = json.load(f)

    m = Phase2aMetrics(n_file_rows=len(test_cases))
    print(f"====== RepoRouter Phase 2A · 基座 · {len(test_cases)} 条 ======\n")

    for _i, case in enumerate(tqdm(test_cases, desc="base")):
        gold = read_gold_answer(case)
        if gold is None:
            m.skipped_bad_gold += 1
            continue
        raw = generate_reply(
            tokenizer,
            model,
            case,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=False,
        )
        pred = parse_routing_json(raw) if raw else None
        m.update(gold, pred)

    for line in m.report_lines():
        print(line)


if __name__ == "__main__":
    main()
