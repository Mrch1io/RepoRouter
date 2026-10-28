import copy
import hashlib
import importlib.util
import io
import json
import tempfile
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletionMessage

import src.agent.core as core
import src.agent.llm as llm
from src.evaluation.stats import bootstrap_ci, mcnemar_p

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def load(path):
    spec = importlib.util.spec_from_file_location("eval_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NEW = load(HERE.parent / "source/scripts/eval_agent.py")
JUDGE = load(HERE.parent / "source/scripts/judge_answers.py")
OLD_JUDGE = load(HERE / "judge_answers_before.py")


@pytest.fixture
def work():
    with tempfile.TemporaryDirectory(dir=HERE) as directory:
        yield Path(directory)


def run_eval(module, backend, scenario, path):
    import src.agent.router as router

    requests = []
    tools = []
    route_calls = []
    step = 0
    wants_tool = scenario in {"search", "tool_error", "answer_error"}
    arguments = {"query": "RepoAgent actual return shape", "top_k": 8}
    decision = {"decision_type": "tool_call", "tool_name": "search_codebase", "arguments": arguments} if wants_tool else {"decision_type": "no_tool"}
    if scenario == "clarify":
        decision = {"decision_type": "clarify_needed", "clarify_answer": "请补充仓库"}

    def route_create(**kwargs):
        route_calls.append(copy.deepcopy(kwargs))
        if scenario == "router_error":
            raise TimeoutError("router offline")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(decision)))])

    def create(**kwargs):
        nonlocal step
        requests.append(copy.deepcopy(kwargs))
        step += 1
        if scenario == "api_error" or scenario == "answer_error" and (backend == "phase2a" or step == 2):
            raise TimeoutError("answer offline")
        if backend == "native" and wants_tool and step == 1:
            message = ChatCompletionMessage(role="assistant", content=None, tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "search_codebase", "arguments": json.dumps(arguments)}}])
            reason = "tool_calls"
        else:
            message = ChatCompletionMessage(role="assistant", content="完整回答" * 1300)
            reason = "stop"
        usage = SimpleNamespace(prompt_tokens=123, completion_tokens=45, prompt_cache_hit_tokens=100, prompt_cache_miss_tokens=23)
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=message, finish_reason=reason)])

    def search(**kwargs):
        tools.append(copy.deepcopy(kwargs))
        if scenario == "tool_error":
            raise ValueError("retrieval failed")
        return json.dumps({"query": kwargs["query"], "top1_distance": .7, "soft_fuse_triggered": False, "results": [{"source": "src/config.py", "content": "证据内容" * 2000}]}, ensure_ascii=False)

    manifest = [{"id": "p02", "query": "问题", "bucket_type": "domain", "grounding_tier": "must_ground" if wants_tool else "ok_generic", "should_call_tool": "yes" if wants_tool else "no", "should_hit_private_knowledge": "yes" if wants_tool else "no", "should_soft_fallback_if_searched": "no"}]
    with ExitStack() as stack:
        stack.enter_context(patch.object(module, "build_e2e_manifest", return_value=manifest))
        stack.enter_context(patch.object(module, "perf_counter", return_value=10.0))
        stack.enter_context(patch.object(core.time, "perf_counter", return_value=10.0))
        stack.enter_context(patch("src.agent.tools.search_tool._get_vector_store", return_value=None))
        stack.enter_context(patch.object(core, "search_codebase", side_effect=search))
        stack.enter_context(patch.object(llm.client.chat.completions, "create", side_effect=create))
        stack.enter_context(patch.object(router, "_client", SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=route_create)))))
        stack.enter_context(patch("sys.argv", ["eval_agent.py", "--routing-backend", backend, "--model", "deepseek-flash", "--max-tokens", "4096", "--jsonl-out", str(path)]))
        stack.enter_context(redirect_stdout(io.StringIO()))
        module.main()
    return json.loads(path.read_text(encoding="utf-8")), requests, tools, route_calls


@pytest.mark.parametrize("backend,scenario", [("native", s) for s in ("direct", "search", "tool_error", "api_error", "answer_error")] + [("phase2a", s) for s in ("direct", "search", "tool_error", "api_error", "answer_error", "router_error", "clarify")])
def test_live_paths_frozen_and_usage(backend, scenario, work):
    old, *old_calls = run_eval(load(HERE / "eval_agent_before.py"), backend, scenario, work / "old.jsonl")
    new, *new_calls = run_eval(NEW, backend, scenario, work / "new.jsonl")
    assert old_calls == new_calls
    assert {k: v for k, v in new.items() if k not in {"usage", "search_arguments", "tool_actions"}} == old
    if "error" in scenario and scenario != "tool_error":
        assert new["usage"] is None
        assert new["execution_error"]
    else:
        usage = new["usage"]
        count = 0 if scenario == "clarify" else 2 if backend == "native" and scenario in {"search", "tool_error"} else 1
        assert usage == {"event": "usage", "calls": count, "cache_hit": 100 * count, "cache_miss": 23 * count, "output": 45 * count, "cost_usd": pytest.approx(count * (100*.006 + 23*.3 + 45*1.2)/1e6), "seconds": 0.0}
    assert new["search_arguments"] == ([{"query": "RepoAgent actual return shape", "top_k": 8}] if new["actual_call_search_codebase"] else [])
    assert [action["arguments"] for action in new["tool_actions"] if action["tool_name"] == "search_codebase"] == new["search_arguments"]


def row(sid, searched=True, chars=10, distance=.7, seconds=2, correct=True, usage=None):
    return {"sample_id": sid, "repeat": 0, "question": sid, "grounding_tier": "must_ground", "actual_call_search_codebase": searched, "top1_distance": distance if searched else None, "tool_contexts": [{"tool_name": "search_codebase", "content": json.dumps({"query": "query", "top1_distance": distance, "results": [{"source": "x.py", "content": "中" * chars}]})}] if searched else [], "search_arguments": [{"query": "query"}] if searched else [], "l1_correct": correct, "seconds": seconds, "cost_usd": .01, "final_answer": "abc", "usage": usage, "execution_error": None}


def write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def compare(a, b, **kwargs):
    with redirect_stdout(io.StringIO()):
        return NEW._compare(a, b, **kwargs)


def test_metrics_are_paired_continuous_and_binary(work):
    a = write(work / "a.jsonl", [row("p02", chars=10, distance=.4, seconds=2), row("x01", chars=20, distance=.5, seconds=3, correct=False), row("p04", chars=30, distance=.6, seconds=4)])
    b = write(work / "b.jsonl", [row("p04", searched=False, seconds=7, correct=False), row("p02", chars=6, distance=.7, seconds=3), row("x01", chars=16, distance=.8, seconds=5)])
    ja = write(work / "ja.jsonl", [{"sample_id": s, "score": score, "final_judgement": label} for s, score, label in [("p02", 5, "correct"), ("p04", 4, "partially_correct"), ("x01", 2, "incorrect")]])
    jb = write(work / "jb.jsonl", [{"sample_id": s, "score": score, "final_judgement": label} for s, score, label in [("x01", 4, "correct"), ("p04", 3, "partially_correct"), ("p02", 5, "correct")]])
    plan = work / "plan.json"
    plan.write_text(json.dumps({"random_sample_ids": ["x01", "p04"], "seed": 20261024}), encoding="utf-8")
    result = compare(a, b, judges=[ja, jb], review_plan=plan, out=work / "out.json")
    summary = result["subsets"]["all"]
    assert summary["n"] == summary["must_ground_n"] == 3
    assert result["subsets"]["known_pilot22"]["n"] == 2
    assert result["subsets"]["remaining68"]["n"] == 1
    assert summary["metrics"]["seconds"]["mean_delta_B_minus_A"] == 2
    assert summary["metrics"]["seconds"]["delta_ci95"] == bootstrap_ci([1, 3, 2])
    assert summary["metrics"]["top1_distance"]["n"] == 2
    assert summary["metrics"]["top1_distance"]["mean_delta_B_minus_A"] == pytest.approx(.3)
    assert summary["metrics"]["evidence_chars"]["n"] == 3
    assert summary["metrics"]["evidence_chars"]["mean_delta_B_minus_A"] == pytest.approx(-38/3)
    assert summary["metrics"]["usage_output"]["n"] == 0
    assert summary["binary"]["l1_correct"] == {"overall_n": 3, "n": 3, "A_correct": 2, "B_correct": 2, "A_only": 1, "B_only": 1, "mcnemar_p": mcnemar_p(1, 1)}
    assert summary["binary"]["final_judgement"]["A_only"] == 0
    assert summary["binary"]["final_judgement"]["B_only"] == 1
    assert all("mcnemar_p" not in metric for metric in summary["metrics"].values())
    assert result["pairs"][0]["A"]["retrieval"]["calls"][0]["top_k"] == 6
    assert result["pairs"][1]["B"]["retrieval"] == {"status": "not_searched", "top1_distance": None, "evidence_chars": 0, "results_count": 0, "calls": [], "non_payload_messages": 0}
    assert result["review_queue"] == [{"sample_id": "p04", "repeat": 0, "selection": ["random"], "human_review": "pending"}, {"sample_id": "x01", "repeat": 0, "selection": ["random", "judgement_disagreement"], "human_review": "pending"}]
    with pytest.raises(FileExistsError):
        compare(a, b, out=work / "out.json")


@pytest.mark.parametrize("problem", ["duplicate_answer", "missing_answer", "empty_answer", "duplicate_judge", "missing_judge", "judge_bad_label", "judge_bad_score", "judge_bool_score", "judge_nan_score", "judge_wrong_hash", "different_question", "different_tier", "random_duplicate", "random_unknown", "both_missing_same_id"])
def test_invalid_comparisons_fail_instead_of_dropping_rows(problem, work):
    answers = [row("p02"), row("p04")]
    second = copy.deepcopy(answers)
    judges = [{"sample_id": r["sample_id"], "score": 4, "final_judgement": "correct"} for r in answers]
    ja_rows = copy.deepcopy(judges)
    plan = {"random_sample_ids": ["p02"]}
    if problem == "duplicate_answer": second.append(second[0])
    if problem == "missing_answer": second.pop()
    if problem == "empty_answer": second.clear()
    if problem == "duplicate_judge": ja_rows.append(ja_rows[0])
    if problem == "missing_judge": ja_rows.pop()
    if problem == "judge_bad_label": ja_rows[0]["final_judgement"] = "pending"
    if problem == "judge_bad_score": ja_rows[0]["score"] = 6
    if problem == "judge_bool_score": ja_rows[0]["score"] = True
    if problem == "judge_nan_score": ja_rows[0]["score"] = float("nan")
    if problem == "judge_wrong_hash": ja_rows[0]["input_sha256"] = "wrong"
    if problem == "different_question": second[0]["question"] = "different"
    if problem == "different_tier": second[0]["grounding_tier"] = "ok_generic"
    if problem == "random_duplicate": plan["random_sample_ids"] *= 2
    if problem == "random_unknown": plan["random_sample_ids"] = ["absent"]
    if problem == "both_missing_same_id": plan["population_ids"] = ["p02", "p04", "p11"]
    a, b = write(work / "a.jsonl", answers), write(work / "b.jsonl", second)
    ja, jb = write(work / "ja.jsonl", ja_rows), write(work / "jb.jsonl", judges)
    plan_path = work / "review.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError):
        compare(a, b, judges=[ja, jb], review_plan=plan_path)


def test_retrieval_context_accounting_and_absent_usage(work):
    record = row("p02", chars=9)
    record["search_arguments"][0]["top_k"] = 8
    assert NEW._retrieval_metrics(record)["calls"][0]["top_k"] == 8
    assert NEW._retrieval_metrics(record)["evidence_chars"] == 9
    record["tool_contexts"].append(copy.deepcopy(record["tool_contexts"][0]))
    assert NEW._retrieval_metrics(record)["evidence_chars"] == 18
    assert NEW._retrieval_metrics(record)["results_count"] == 2
    record["tool_contexts"] = []
    assert NEW._retrieval_metrics(record)["status"] == "missing_context"
    assert NEW._retrieval_metrics(record)["evidence_chars"] is None
    assert NEW._retrieval_metrics(record)["top1_distance"] is None
    for content in ("not json", "[]", '{"results": null}'):
        record["tool_contexts"] = [{"tool_name": "search_codebase", "content": content}]
        assert NEW._retrieval_metrics(record)["status"] == "invalid_context"
        assert NEW._retrieval_metrics(record)["evidence_chars"] is None
        assert NEW._retrieval_metrics(record)["top1_distance"] is None
    a = write(work / "a.jsonl", [row("p02", usage={"calls": 0, "output": 0})])
    b = write(work / "b.jsonl", [row("p02", usage={"calls": 1, "output": 5})])
    result = compare(a, b)
    assert result["subsets"]["all"]["metrics"]["usage_output"]["n"] == 1
    assert result["subsets"]["all"]["metrics"]["usage_output"]["mean_delta_B_minus_A"] == 5
    assert result["subsets"]["all"]["binary"]["final_judgement"]["n"] == 0


@pytest.mark.parametrize("explicit", [True, False])
def test_judge_rejects_duplicate_first_repeat_before_any_request(work, explicit):
    record = row("p02")
    if not explicit:
        record.pop("repeat")
    source = write(work / "duplicate.jsonl", [record, record])
    with patch.object(JUDGE._judge_client.chat.completions, "create") as request:
        with pytest.raises(ValueError, match="重复的 repeat=0"):
            JUDGE._load_jsonl_by_id(source)
        request.assert_not_called()


def test_judge_later_repeats_still_ignored(work):
    first = row("p02")
    later = dict(first, repeat=1, final_answer="第二次回答")
    source = write(work / "repeats.jsonl", [later, first, later])
    assert JUDGE._load_jsonl_by_id(source) == OLD_JUDGE._load_jsonl_by_id(source) == {"p02": first}


@pytest.mark.parametrize("scenario", ["success", "answer_error", "judge_error"])
def test_judge_actual_request_and_output_frozen(work, scenario):
    record = row("p02")
    record["answer_finish_reason"] = "stop"
    if scenario == "answer_error":
        record["execution_error"] = "TimeoutError: answer failed"
    source = write(work / "answer.jsonl", [record])
    results = []
    requests = []
    for index, module in enumerate((OLD_JUDGE, JUDGE)):
        captured = []
        output = work / f"judged_{index}.jsonl"

        def create(**kwargs):
            captured.append(copy.deepcopy(kwargs))
            if scenario == "judge_error":
                raise TimeoutError("judge failed")
            value = {"answer_relevant": True, "grounding_appropriate": True, "context_pollution": False, "score": 4, "final_judgement": "correct", "reason": "评判理由"}
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)))])

        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "PHASE1_JUDGE_SAMPLE_IDS", ["p02"]))
            stack.enter_context(patch.object(module._judge_client.chat.completions, "create", side_effect=create))
            stack.enter_context(patch("sys.argv", ["judge_answers.py", "--jsonl", str(source), "--out", str(output), "--workers", "1"]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            module.main()
        requests.append(captured)
        results.append(output.read_text(encoding="utf-8"))
    assert requests[0] == requests[1]
    assert results[0] == results[1]
    assert len(requests[0]) == (0 if scenario == "answer_error" else 1)


def test_judge_failures_retained_and_truncation_flagged(work):
    answers = [row("p02"), row("p04")]
    second = copy.deepcopy(answers)
    second[1]["answer_finish_reason"] = "length"
    a, b = write(work / "a.jsonl", answers), write(work / "b.jsonl", second)
    ja = write(work / "ja.jsonl", [{"sample_id": "p02", "error_stage": "judge_request", "error": "timeout", "score": 5, "final_judgement": "correct"}, {"sample_id": "p04", "score": 4, "final_judgement": "correct"}])
    jb = write(work / "jb.jsonl", [{"sample_id": s, "score": 4, "final_judgement": "correct"} for s in ("p02", "p04")])
    result = compare(a, b, judges=[ja, jb])
    summary = result["subsets"]["all"]
    assert result["publishable"] is False
    assert summary["n"] == 2
    assert summary["judge_errors"] == {"A": 1, "B": 0}
    assert summary["truncated_answers"] == {"A": 0, "B": 1}
    assert summary["binary"]["final_judgement"]["overall_n"] == 2
    assert summary["binary"]["final_judgement"]["n"] == 1
    assert summary["metrics"]["score"]["overall_n"] == 2
    assert summary["metrics"]["score"]["n"] == 1
    assert result["pairs"][0]["A"]["score"] is None
    assert result["pairs"][0]["A"]["final_judgement"] is None
    assert result["pairs"][0]["A"]["judge_error"] == {"stage": "judge_request", "message": "timeout"}
    assert result["review_queue"] == [{"sample_id": "p02", "repeat": 0, "selection": ["judge_error"], "human_review": "pending"}, {"sample_id": "p04", "repeat": 0, "selection": ["truncated"], "human_review": "pending"}]
    failed_rows = [json.loads(line) for line in ja.read_text(encoding="utf-8").splitlines()]
    failed_rows[0]["input_sha256"] = "incorrect"
    write(ja, failed_rows)
    with pytest.raises(ValueError, match="校验和"):
        compare(a, b, judges=[ja, jb])


def test_duplicate_interception_keeps_evidence_and_argument_alignment():
    record = row("p02", chars=9, distance=.4)
    other = row("p02", chars=11, distance=.8)
    payload = json.loads(other["tool_contexts"][0]["content"])
    payload["query"] = "second"
    record["tool_contexts"] = [{"tool_name": "search_codebase", "content": "工具执行失败"}, record["tool_contexts"][0], {"tool_name": "search_codebase", "content": "系统拦截重复工具调用"}, {"tool_name": "search_codebase", "content": json.dumps(payload)}]
    record["search_arguments"] = [{"query": "failed", "top_k": 8}, {"query": "query"}, {"query": "second", "top_k": 4}]
    result = NEW._retrieval_metrics(record)
    assert result["status"] == "searched"
    assert result["non_payload_messages"] == 2
    assert result["evidence_chars"] == 20
    assert result["results_count"] == 2
    assert result["top1_distance"] == .8
    assert [c["top_k"] for c in result["calls"]] == [6, 4]


def test_blind_review_cards_map_back_without_judge_or_chain_leak(work):
    answers = [row("p02"), row("p04")]
    for i, item in enumerate(answers):
        item["final_answer"] = f"A answer {i}"
        item["tool_actions"] = [{"tool_name": "search_codebase", "arguments": {"query": "query"}}]
    second = copy.deepcopy(answers)
    for i, item in enumerate(second):
        item["final_answer"] = f"B answer {i}"
    a, b = write(work / "a.jsonl", answers), write(work / "b.jsonl", second)
    plan = work / "plan.json"
    plan.write_text(json.dumps({"random_sample_ids": ["p02", "p04"], "population_ids": ["p02", "p04"], "seed": 19}), encoding="utf-8")
    outputs = []
    for name in ("first", "second"):
        result = compare(a, b, review_plan=plan, out=work / f"{name}.json")
        cards = [json.loads(line) for line in Path(result["review_files"]["blinded"]).read_text(encoding="utf-8").splitlines()]
        mapping = json.loads(Path(result["review_files"]["mapping"]).read_text(encoding="utf-8"))
        assert len(cards) == len(mapping) == 4
        for card, entry in zip(cards, mapping):
            assert card["review_id"] == entry["review_id"]
            assert not {"chain", "sample_id", "score", "final_judgement", "selection", "judge_model", "routing_backend"} & card.keys()
            source = {r["sample_id"]: r for r in (answers if entry["chain"] == "A" else second)}[entry["sample_id"]]
            assert card["final_answer"] == source["final_answer"]
            assert card["actual_tool_actions"] == source["tool_actions"]
            assert card["tool_contexts"] == source["tool_contexts"]
            assert entry["selection"] == ["random"]
        outputs.append((cards, mapping))
    assert outputs[0] == outputs[1]
    blocked = work / "blocked.review-map.json"
    blocked.write_text("keep existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        compare(a, b, review_plan=plan, out=work / "blocked.json")
    assert blocked.read_text(encoding="utf-8") == "keep existing"
    assert not (work / "blocked.json").exists()
    assert not (work / "blocked.review.jsonl").exists()
