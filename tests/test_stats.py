#统计小工具：用手算过的例子核对
import json
import sys

import pytest

from scripts import eval_agent,report_agent_errors,rescore_agent
from src.evaluation.stats import wilson_ci,bootstrap_ci,mcnemar_p,roc_auc,cohen_kappa


@pytest.mark.parametrize("actual,expect,result", [
    (True,"yes",(True,"none")), (False,"yes",(False,"fn")),
    (True,"no",(False,"fp")), (False,"no",(True,"none")),
    (True,"optional",(True,"none")), (False,"optional",(True,"none")),
])
def test_tool_use_scoring(actual,expect,result):
    assert report_agent_errors._score_tool_use(actual,expect) == result
    assert rescore_agent._score_tool_use(actual,expect) == result


@pytest.mark.parametrize("expect", ["unknown",None])
def test_tool_use_scoring_keeps_invalid_label_errors(expect):
    #1.两个脚本原来的报错文字不同，复用评分逻辑时也要保留
    with pytest.raises(ValueError) as report_error:
        report_agent_errors._score_tool_use(False,expect)
    with pytest.raises(ValueError) as rescore_error:
        rescore_agent._score_tool_use(False,expect)
    assert report_error.value.args == (expect,)
    assert rescore_error.value.args == (f"unknown expect: {expect}",)


def test_rescore_and_report_keep_failures_out_of_fp_fn(tmp_path,monkeypatch):
    #1.覆盖三种标签的两种决策，另外放入两条运行失败记录
    labels = ("yes","no","optional")
    manifest = [{"id": label,"should_call_tool": label,"bucket_type": "domain",
                 "grounding_tier": "must_ground" if label == "yes" else "prefer_ground"}
                for label in labels]
    rows = [{"sample_id": label,"repeat": repeat,"actual_call_search_codebase": actual,
             "bucket_type": "domain"}
            for label in labels for repeat,actual in enumerate((False,True))]
    rows.extend({"sample_id": label,"repeat": 2,"actual_call_search_codebase": actual,
                 "bucket_type": "domain","execution_error": "连接中断"}
                for label,actual in (("yes",False),("no",True)))
    path = tmp_path / "answers.jsonl"
    path.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")
    monkeypatch.setattr(report_agent_errors,"build_e2e_manifest",lambda: manifest)
    monkeypatch.setattr(rescore_agent,"build_e2e_manifest",lambda: manifest)

    #2.统计和错例报告必须对应同一题、同一轮，运行失败单独计数
    metrics = rescore_agent.compute_metrics(path)
    report = report_agent_errors.analyze(path)
    assert (metrics.n,metrics.l1_ok,metrics.l1_fp,metrics.l1_fn,metrics.execution_errors) == (8,4,1,1,2)
    assert (metrics.must_n,metrics.must_ok,metrics.must_fn) == (3,1,1)
    assert metrics.fp_rows == [("no",1,"domain","")]
    assert metrics.fn_rows == [("yes",0,"domain","")]
    assert report["n"] == 8 and len(report["execution_errors"]) == 2
    assert [(item["rec"]["sample_id"],item["rec"]["repeat"]) for item in report["fp"]] == [("no",1)]
    assert [(item["rec"]["sample_id"],item["rec"]["repeat"]) for item in report["fn"]] == [("yes",0)]


def test_stats():
    lo,hi = wilson_ci(88,90)
    assert 0.92 < lo < 0.93 and 0.99 < hi < 1.0
    assert wilson_ci(0,0) == (0.0,0.0)
    lo,hi = bootstrap_ci([1.0]*10)
    assert lo == hi == 1.0
    assert abs(mcnemar_p(10,2) - 0.0386) < 0.001   #二项分布精确值
    assert mcnemar_p(3,3) == 1.0
    assert roc_auc([2,3,4],[1,2]) == 5.5 / 6
    #一致率 0.8、碰巧一致 0.52 -> kappa = (0.8-0.52)/(1-0.52)
    a = ["y"]*6 + ["n"]*4
    b = ["y"]*5 + ["n"] + ["n"]*3 + ["y"]
    assert abs(cohen_kappa(a,b) - (0.8 - 0.52) / 0.48) < 1e-9
    assert cohen_kappa(a,a) == 1.0


def _compare_rows(tmp_path, cases, name, judges=False):
    #1.只在临时目录造配对记录，不读取或覆盖历史成绩
    paths, judge_paths = [], []
    for side in range(2):
        rows, scores = [], []
        for sample_id, repeat, correct, seconds in cases:
            rows.append({"sample_id": sample_id, "repeat": repeat, "question": sample_id,
                         "grounding_tier": "must_ground", "l1_correct": correct[side],
                         "actual_call_search_codebase": False, "seconds": seconds[side],
                         "final_answer": "测试回答"})
            scores.append({"sample_id": sample_id, "repeat": repeat,
                           "answer_format": "full", "judge_gold_mode": "legacy",
                           "judge_model": "test-judge", "judge_prompt_version": "full_answer_v1",
                           "judge_temperature": 0.1, "judge_max_tokens": 1200,
                           **({"score": 5 if correct[side] else 1,
                               "final_judgement": "correct" if correct[side] else "incorrect"}
                              if seconds[side] is not None else {"error": "评分缺失"})})
        path = tmp_path / f"{name}-{side}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        paths.append(path)
        if judges:
            path = tmp_path / f"{name}-{side}-judge.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in scores), encoding="utf-8")
            judge_paths.append(path)
    return eval_agent._compare(*paths, judges=judge_paths or None)["subsets"]["all"]


def test_repeated_comparison_keeps_question_as_independent_unit(tmp_path):
    #1.同样10题复制3次，不能把不显著的差异变成显著，也不能缩小区间
    cases = [(f"q{i}", 0, (i < 7, i >= 7), (1, i + 1)) for i in range(10)]
    once = _compare_rows(tmp_path, cases, "once", judges=True)
    repeated = _compare_rows(tmp_path, [(sid, rep, correct, secs)
                                      for rep in range(3) for sid, _, correct, secs in cases], "repeated", judges=True)
    assert (once["n"], repeated["n"], repeated["n_questions"]) == (10, 30, 10)
    for field in ("seconds", "score"):
        a, b = once["metrics"][field], repeated["metrics"][field]
        assert (a["n"], b["n"], b["n_questions"]) == (10, 30, 10)
        for key in ("A_mean", "B_mean", "mean_delta_B_minus_A", "delta_ci95"):
            assert a[key] == b[key]
    assert once["metrics"]["seconds"]["delta_ci95"] == bootstrap_ci(list(range(10)))
    assert once["metrics"]["seconds"]["mean_delta_B_minus_A"] == 4.5
    for field in ("l1_correct", "final_judgement"):
        a, b = once["binary"][field], repeated["binary"][field]
        assert (a["A_only"], a["B_only"], a["mcnemar_p"]) == (7, 3, 0.34375)
        assert (b["A_only"], b["B_only"], b["n_questions"]) == (21, 9, 10)
        assert b["mcnemar_p"] is None
        assert [row["mcnemar_p"] for row in b["by_repeat"]] == [0.34375] * 3


def test_repeated_comparison_weights_questions_equally_and_skips_missing_pairs(tmp_path):
    #1.重复次数不齐也让每题权重相同；只用两边都有值的配对
    cases = [("q1", 0, (True, False), (0, 10)), ("q1", 1, (True, False), (0, 10)),
             ("q2", 0, (False, True), (0, 0)), ("q3", 1, (True, True), (None, 999))]
    result = _compare_rows(tmp_path, cases, "missing", judges=True)
    seconds = result["metrics"]["seconds"]
    assert (result["n"], result["n_questions"], seconds["n"], seconds["n_questions"]) == (4, 3, 3, 2)
    assert (seconds["A_mean"], seconds["B_mean"], seconds["mean_delta_B_minus_A"]) == (0, 5, 5)
    assert seconds["delta_ci95"] == bootstrap_ci([10, 0])
    judged = result["binary"]["final_judgement"]
    assert (judged["n"], judged["n_questions"], judged["mcnemar_p"]) == (3, 2, None)
    assert [row["n"] for row in judged["by_repeat"]] == [2, 1]
    assert result["metrics"]["cost_usd"]["n_questions"] == 0
    assert result["metrics"]["cost_usd"]["delta_ci95"] is None


def test_judge_missing_whole_repeat_does_not_report_pooled_p(tmp_path):
    result = _compare_rows(tmp_path, [("q1", 0, (True, False), (1, 2)),
                                    ("q1", 1, (True, False), (None, 2))], "missing-round", judges=True)
    judged = result["binary"]["final_judgement"]
    assert judged["mcnemar_p"] is None
    assert [row["mcnemar_p"] for row in judged["by_repeat"]] == [1.0, None]


@pytest.mark.parametrize("repeats", [1, 3])
def test_eval_summary_wilson_intervals_use_one_round(tmp_path, monkeypatch, capsys, repeats):
    #1.直接走评测入口，但用固定回答替身，确保不访问模型或真实评测数据
    class Agent:
        def clear_memory(self):
            pass

        def chat(self, question, trace, max_tokens):
            if question == "private":
                trace.append({"tool_name": "search_codebase", "arguments": {"query": question},
                              "search_payload": {"top1_distance": 0.1, "soft_fuse_triggered": False}})
            return "测试回答"

    manifest = [{"id": q, "query": q, "bucket_type": "domain", "grounding_tier": tier,
                 "should_call_tool": call, "should_hit_private_knowledge": call,
                 "should_soft_fallback_if_searched": "no"}
                for q, tier, call in (("private", "must_ground", "yes"), ("general", "n/a", "no"))]
    calls = []

    def record_wilson(k, n):
        calls.append((k, n))
        return wilson_ci(k, n)

    monkeypatch.setattr(eval_agent, "RepoRouter", Agent)
    monkeypatch.setattr(eval_agent, "build_e2e_manifest", lambda: manifest)
    monkeypatch.setattr(eval_agent, "wilson_ci", record_wilson)
    monkeypatch.setattr(sys, "argv", ["eval_agent", "--repeat", str(repeats), "--routing-backend", "native",
                                     "--jsonl-out", str(tmp_path / "eval.jsonl")])
    eval_agent.main()
    assert calls == [(2, 2)] * repeats + [(1, 1)] * (2 * repeats)
    output = capsys.readouterr().out
    assert ("重复观测汇总" in output) == (repeats > 1)
    if repeats == 1:
        assert "95%CI [34.2, 100.0]" in output


@pytest.mark.parametrize("field,value", [
    ("answer_format","legacy_excerpt"), ("judge_gold_mode","index_aware"),
    ("judge_model","other-judge"), ("judge_prompt_version","different_prompt"),
    ("judge_temperature",0.9), ("judge_max_tokens",500),
])
@pytest.mark.parametrize("within_file", [False,True])
def test_compare_rejects_mixed_judge_protocol(tmp_path,field,value,within_file):
    #1.同一批题的评分条件要一致，文件之间和文件内部都检查
    cases = [("q1",0,(True,True),(1,2)), ("q2",0,(True,True),(1,2))]
    _compare_rows(tmp_path,cases,"mixed",judges=True)
    paths = [tmp_path / f"mixed-{side}.jsonl" for side in range(2)]
    judges = [tmp_path / f"mixed-{side}-judge.jsonl" for side in range(2)]
    target = judges[0] if within_file else judges[1]
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    for row in (rows[1:] if within_file else rows):
        row[field] = value
    target.write_text("".join(json.dumps(row)+"\n" for row in rows),encoding="utf-8")
    out = tmp_path / "comparison.json"
    with pytest.raises(ValueError,match="评分口径|完整回答"):
        eval_agent._compare(*paths,judges=judges,out=out)
    assert not out.exists()


@pytest.mark.parametrize("field", ["answer_format","judge_gold_mode","judge_model",
                                   "judge_prompt_version","judge_temperature","judge_max_tokens"])
def test_compare_rejects_missing_judge_protocol(tmp_path,field):
    #1.两边都缺字段也不能默认当作使用了相同的评分口径
    _compare_rows(tmp_path,[("q1",0,(True,True),(1,2))],"missing-protocol",judges=True)
    paths = [tmp_path / f"missing-protocol-{side}.jsonl" for side in range(2)]
    judges = [tmp_path / f"missing-protocol-{side}-judge.jsonl" for side in range(2)]
    for target in judges:
        row = json.loads(target.read_text(encoding="utf-8"))
        row.pop(field)
        target.write_text(json.dumps(row)+"\n",encoding="utf-8")
    with pytest.raises(ValueError,match="评分口径|完整回答"):
        eval_agent._compare(*paths,judges=judges)


@pytest.mark.parametrize("answer", [None,""])
def test_compare_checks_actual_full_answer(tmp_path,answer):
    #1.评分虽然写full，实际只有预览或空文本时仍不能当作完整回答成绩
    _compare_rows(tmp_path,[("q1",0,(True,True),(1,2))],"answer-format",judges=True)
    paths = [tmp_path / f"answer-format-{side}.jsonl" for side in range(2)]
    judges = [tmp_path / f"answer-format-{side}-judge.jsonl" for side in range(2)]
    row = json.loads(paths[0].read_text(encoding="utf-8"))
    if answer is None:
        row["final_answer_preview"] = row.pop("final_answer")
    else:
        row["final_answer"] = answer
    paths[0].write_text(json.dumps(row)+"\n",encoding="utf-8")
    with pytest.raises(ValueError,match="完整回答"):
        eval_agent._compare(*paths,judges=judges)


@pytest.mark.parametrize("gold_mode", ["legacy","index_aware"])
def test_compare_accepts_matching_protocol_and_runtime_only(tmp_path,gold_mode):
    #1.相同口径的完整评分可比较，只有运行记录时不强求评分元数据
    _compare_rows(tmp_path,[("q1",0,(True,False),(1,2))],"matching",judges=True)
    paths = [tmp_path / f"matching-{side}.jsonl" for side in range(2)]
    judges = [tmp_path / f"matching-{side}-judge.jsonl" for side in range(2)]
    for target in judges:
        row = json.loads(target.read_text(encoding="utf-8"))
        row["judge_gold_mode"] = gold_mode
        target.write_text(json.dumps(row)+"\n",encoding="utf-8")
    result = eval_agent._compare(*paths,judges=judges)
    assert result["publishable"] is True
    assert result["subsets"]["all"]["metrics"]["score"]["mean_delta_B_minus_A"] == -4
    assert eval_agent._compare(*paths)["publishable"] is False
