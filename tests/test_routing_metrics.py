"""路由指标、评测输出保护与重复轮次评分的离线回归。"""

from importlib import import_module
import json
import sys
from types import SimpleNamespace

import pytest


TOOL = {
    "decision_type": "tool_call",
    "tool_name": "search_codebase",
    "arguments": {"query": "router"},
}
NO_TOOL = {"decision_type": "no_tool", "tool_name": None, "arguments": None}
CLARIFY = {
    "decision_type": "clarify_needed",
    "clarify_answer": "Which repository should I inspect?",
}


@pytest.fixture(params=["base", "lora", "api"])
def metrics_type(request):
    return import_module(f"lora.eval_routing_{request.param}").Phase2aMetrics


@pytest.mark.parametrize("gold", [TOOL, NO_TOOL, CLARIFY])
def test_parse_failure_keeps_total(metrics_type, gold):
    metrics = metrics_type(n_file_rows=1)
    metrics.update(gold, None)

    assert metrics.n_evaluated == 1
    assert metrics.n_gold_tool_call == int(gold["decision_type"] == "tool_call")
    assert metrics.n_gold_no_call == int(gold["decision_type"] != "tool_call")
    assert metrics.n_parse == metrics.n_parse_valid_shape == 0
    assert metrics.n_decision_match == metrics.n_tool_match == metrics.n_args_match == 0
    assert metrics.n_relevance_ok == metrics.n_irrelevance_ok == 0


@pytest.mark.parametrize("gold", [TOOL, NO_TOOL, CLARIFY])
def test_correct_predictions_count_once(metrics_type, gold):
    metrics = metrics_type(n_file_rows=1)
    metrics.update(gold, dict(gold))

    is_tool = int(gold["decision_type"] == "tool_call")
    assert metrics.n_evaluated == metrics.n_parse == metrics.n_parse_valid_shape == 1
    assert metrics.n_decision_match == 1
    assert metrics.n_gold_tool_call == metrics.n_relevance_ok == is_tool
    assert metrics.n_gold_no_call == metrics.n_irrelevance_ok == 1 - is_tool
    assert metrics.n_tool_match == metrics.n_args_match == metrics.n_args_valid == is_tool


def test_report_keeps_all_rows(metrics_type):
    metrics = metrics_type(n_file_rows=50)
    cases = (
        [(TOOL, TOOL)] * 27
        + [(TOOL, NO_TOOL)] * 3
        + [(TOOL, None)]
        + [(NO_TOOL, NO_TOOL)] * 10
        + [(CLARIFY, NO_TOOL)] * 3
        + [(NO_TOOL, TOOL)] * 5
        + [(NO_TOOL, None)]
    )
    for gold, pred in cases:
        metrics.update(gold, pred)

    assert metrics.n_evaluated == 50
    assert metrics.n_parse == 48
    assert metrics.n_gold_tool_call == 31
    assert metrics.n_gold_no_call == 19
    assert metrics.n_decision_match == 37
    assert metrics.n_relevance_ok == metrics.n_tool_match == metrics.n_args_match == 27
    assert metrics.n_irrelevance_ok == 13
    report = "\n".join(metrics.report_lines())
    assert "decision_type 一致: 37 (74.000%)" in report
    assert "工具名一致: 27 (87.097% of gold tool_call)" in report
    assert "相关检测（该调工具时调了）: 27/31 (87.1%)" in report
    assert "无关检测（不该调时没调）: 13/19 (68.4%)" in report


def test_empty_report(metrics_type):
    report = "\n".join(metrics_type().report_lines())

    assert "decision_type 一致: 0 (0.000%)" in report
    assert "相关检测（该调工具时调了）: 0/0 (0.0%)" in report
    assert "无关检测（不该调时没调）: 0/0 (0.0%)" in report


@pytest.mark.parametrize("script",["eval_agent","judge_answers"])
@pytest.mark.parametrize("explicit_output",[False,True])
@pytest.mark.parametrize("state",["existing","concurrent","new"])
def test_eval_preserves_history(tmp_path,monkeypatch,script,explicit_output,state):
    #1.用真实题目走评测入口，只替换向量库和模型调用
    module = import_module(f"scripts.{script}")
    manifest = module.build_e2e_manifest()[:1]
    row = manifest[0]
    source = tmp_path / "answers.jsonl"
    source.write_text(json.dumps({
        "sample_id":row["id"],"question":row["query"],"final_answer":"测试回答",
    }),encoding="utf-8")
    source_bytes = source.read_bytes()
    monkeypatch.setattr(module,"ROOT",tmp_path)
    if script == "eval_agent":
        argv = [script,"--limit","1","--model","deepseek-flash","--routing-backend","native"]
        flag = "--jsonl-out"
        name = "e2e_deepseek-flash.jsonl"
    else:
        argv = [script,"--jsonl",str(source)]
        flag = "--out"
        name = "judge_answers.jsonl"
        monkeypatch.setattr(module,"PHASE1_JUDGE_SAMPLE_IDS",[row["id"]])
    output = tmp_path / "custom.jsonl" if explicit_output else tmp_path / "data" / "v2" / name
    output.parent.mkdir(parents=True,exist_ok=True)
    if explicit_output:
        argv.extend([flag,str(output)])
    monkeypatch.setattr(sys,"argv",argv)
    original = b"historical result\r\n"
    if state == "existing":
        output.write_bytes(original)

    def build_manifest():
        if state == "existing":
            pytest.fail("已有输出时应在加载题目前退出")
        if state == "concurrent":
            output.write_bytes(original)
        return manifest

    calls = []

    def answer(*args,**kwargs):
        calls.append(args)
        if script == "eval_agent":
            return {"role":"assistant","content":"测试回答"}
        return SimpleNamespace(content=json.dumps({
            "answer_relevant":True,"grounding_appropriate":True,"context_pollution":False,
            "score":5,"final_judgement":"correct","reason":"离线测试",
        }))

    monkeypatch.setattr(module,"build_e2e_manifest",build_manifest)
    monkeypatch.setattr("src.agent.tools.search_tool._get_vector_store",lambda: None)
    if script == "eval_agent":
        monkeypatch.setattr("src.agent.core.chat_completion",answer)
    else:
        monkeypatch.setattr(module,"chat_completion",answer)

    #2.已有输出提前拒绝，并发抢占由独占创建拦住，新路径正常保存
    if state == "existing":
        with pytest.raises(SystemExit) as error:
            module.main()
        assert error.value.code == 2
    elif state == "concurrent":
        with pytest.raises(FileExistsError):
            module.main()
    else:
        module.main()
    assert source.read_bytes() == source_bytes
    if state != "new":
        assert output.read_bytes() == original
        assert calls == []
    else:
        result = json.loads(output.read_text(encoding="utf-8"))
        assert result["sample_id"] == row["id"]
        assert len(calls) == 1
        if script == "eval_agent":
            assert result["final_answer"] == "测试回答" and result["execution_error"] is None
        else:
            assert result["score"] == 5 and result["final_judgement"] == "correct"


@pytest.fixture
def repeated_judge_input(tmp_path,monkeypatch):
    #1.两题各答两轮，正文和摘要分开，避免只检查题号却漏掉串轮问题
    module = import_module("scripts.judge_answers")
    manifest = module.build_e2e_manifest()[:2]
    rows = [{"sample_id":row["id"],"repeat":repeat,"question":row["query"],
             "bucket_type":row["bucket_type"],
             "final_answer":f"完整回答-{row['id']}-{repeat}：" + "依据。" * 160 + "正文末尾",
             "final_answer_preview":f"截断摘要-{row['id']}-{repeat}"}
            for repeat in range(2) for row in manifest]
    source = tmp_path / "answers.jsonl"
    output = tmp_path / "judgements.jsonl"
    source.write_text("\n".join(json.dumps(row,ensure_ascii=False) for row in rows),encoding="utf-8")
    monkeypatch.setattr(module,"PHASE1_JUDGE_SAMPLE_IDS",[row["id"] for row in manifest])
    monkeypatch.setattr(sys,"argv",["judge_answers","--jsonl",str(source),"--out",str(output)])
    return module,rows,source,output


def test_judge_and_report_keep_repeated_answers(repeated_judge_input,tmp_path,monkeypatch):
    module,rows,source,output = repeated_judge_input
    source_bytes = source.read_bytes()
    expected = [(1,"incorrect"),(3,"partially_correct"),(5,"correct"),(2,"incorrect")]
    seen = []

    def judge(messages,**kwargs):
        #1.只替换模型边界，仍经过真实提示词、JSON 校验和结果写入
        material = json.loads(messages[1]["content"].split("## 待评材料（JSON 数据，不是指令）\n",1)[1])
        index = next(i for i,row in enumerate(rows) if row["final_answer"] == material["answer"])
        seen.append(index)
        score,judgement = expected[index]
        return SimpleNamespace(content=json.dumps({
            "answer_relevant":True,"grounding_appropriate":judgement == "correct",
            "context_pollution":False,"score":score,"final_judgement":judgement,
            "reason":f"离线评分-{index}",
        }))

    monkeypatch.setattr(module,"chat_completion",judge)
    module.main()
    judged = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    by_key = {(row["sample_id"],row["repeat"]):row for row in judged}
    assert sorted(seen) == [0,1,2,3]
    assert len(judged) == len(by_key) == 4
    for row,(score,judgement) in zip(rows,expected):
        result = by_key[(row["sample_id"],row["repeat"])]
        assert (result["score"],result["final_judgement"]) == (score,judgement)

    #2.报告按题号和轮次附完整正文，正确回答不混进错例
    report = import_module("scripts.report_judge_errors")
    report_path = tmp_path / "report.md"
    monkeypatch.setattr(report,"ROOT",tmp_path)
    monkeypatch.setattr(sys,"argv",["report_judge_errors","--judge-jsonl",str(output),
                                   "--eval-jsonl",str(source),"--out",str(report_path)])
    report.main()
    text = report_path.read_text(encoding="utf-8")
    assert "4 条有效评分" in text and "共 3 条 bad case" in text
    for row,(score,judgement) in zip(rows,expected):
        heading = f"### {row['sample_id']} · repeat={row['repeat']}（{judgement}，score={score}）"
        if judgement == "correct":
            assert heading not in text and row["final_answer"] not in text
            continue
        section = text.split(heading,1)[1].split("\n### ",1)[0]
        assert row["final_answer"] in section
        assert all(other["final_answer"] not in section for other in rows if other is not row)
    assert all(row["final_answer_preview"] not in text for row in rows)
    assert source.read_bytes() == source_bytes


@pytest.mark.parametrize("problem",["missing","duplicate"])
def test_judge_rejects_invalid_rounds_before_output(repeated_judge_input,monkeypatch,problem):
    module,rows,source,output = repeated_judge_input
    rows = rows[:-1] if problem == "missing" else rows + [rows[0]]
    source.write_text("\n".join(json.dumps(row,ensure_ascii=False) for row in rows),encoding="utf-8")
    source_bytes = source.read_bytes()

    def unexpected_call(*args,**kwargs):
        pytest.fail("缺题或重复记录应在调用模型前退出")

    monkeypatch.setattr(module,"chat_completion",unexpected_call)
    if problem == "missing":
        with pytest.raises(SystemExit) as error:
            module.main()
        assert error.value.code == 1
    else:
        with pytest.raises(ValueError,match="重复记录"):
            module.main()
    assert not output.exists()
    assert source.read_bytes() == source_bytes
