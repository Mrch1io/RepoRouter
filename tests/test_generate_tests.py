#测试“生成测试后真的跑一遍”
from types import SimpleNamespace

import pytest

import src.agent.tools.test_generation as gt

ADD_CODE = "def add(a,b):\n    return a+b\n"
GOOD_TEST = "from target import add\n\ndef test_add():\n    assert add(1,2) == 3\n"
BAD_TEST = "from target import add\n\ndef test_add():\n    assert add(1,2) == 4\n"


def test_run_pytest_pass_and_fail():
    assert gt.run_pytest(ADD_CODE,GOOD_TEST)[0] is True
    ok,output = gt.run_pytest(ADD_CODE,BAD_TEST)
    assert ok is False and "assert" in output


def test_run_pytest_timeout():
    loop_test = "def test_loop():\n    while True:\n        pass\n"
    ok,output = gt.run_pytest(ADD_CODE,loop_test,timeout=3)
    assert ok is False and "超过" in output


def fake_llm(monkeypatch,replies):
    #假模型：按顺序返回给定的测试代码，并记下每次收到的对话
    prompts = []
    monkeypatch.setattr(gt,"chat_completion",lambda msgs,**kw: prompts.append(list(msgs)) or SimpleNamespace(content=replies.pop(0)))
    return prompts


def test_retry_until_pass(monkeypatch):
    #第一次给错的测试，看到报错后第二次改对
    prompts = fake_llm(monkeypatch,[BAD_TEST,"```python\n" + GOOD_TEST + "```"])

    result = gt.gen_tests(ADD_CODE)
    assert "通过（第 2 次生成），覆盖率 100%" in result
    assert "def add(a,b)" in prompts[0][1]["content"]   #被测代码真的传给了模型
    assert "pytest 输出" in prompts[1][-1]["content"]   #第二次带上了报错


def test_retry_low_coverage(monkeypatch):
    #两个函数只测了一个：能通过，但覆盖率 75% 不到 80%，要求补测试
    code = ADD_CODE + "\ndef sub(a,b):\n    return a-b\n"
    full_test = GOOD_TEST + "\nfrom target import sub\n\ndef test_sub():\n    assert sub(3,1) == 2\n"
    prompts = fake_llm(monkeypatch,[GOOD_TEST,full_test])

    result = gt.gen_tests(code)
    assert "覆盖率只有 75%" in prompts[1][-1]["content"]
    assert "通过（第 2 次生成），覆盖率 100%" in result


#测试没有导入被测代码时，要求补上导入后重新生成
def test_retry_missing_coverage(monkeypatch):
    monkeypatch.setattr(gt,"HAS_COV",True)
    no_import_test = "def test_add():\n    assert 1 + 1 == 2\n"
    prompts = fake_llm(monkeypatch,[no_import_test,GOOD_TEST])

    result = gt.gen_tests(ADD_CODE)
    assert len(prompts) == 2
    assert "没有拿到 target.py 的覆盖率，测试可能没有导入被测代码（应该 from target import ...）" in prompts[1][-1]["content"]
    assert "通过（第 2 次生成）" in result


#连续三次没有覆盖率，必须如实返回失败和原因
def test_missing_coverage_retry_limit(monkeypatch):
    monkeypatch.setattr(gt,"HAS_COV",True)
    monkeypatch.setattr(gt,"run_pytest",lambda *args: (True,"1 passed"))
    prompts = fake_llm(monkeypatch,[GOOD_TEST] * 3)

    result = gt.gen_tests(ADD_CODE)
    assert len(prompts) == 3
    assert "3 次都没通过" in result
    assert "没有拿到 target.py 的覆盖率" in result


#没装pytest-cov时，仍然只看pytest是否通过
def test_pass_without_coverage_plugin(monkeypatch):
    monkeypatch.setattr(gt,"HAS_COV",False)
    monkeypatch.setattr(gt,"run_pytest",lambda *args: (True,"1 passed"))
    prompts = fake_llm(monkeypatch,[GOOD_TEST])

    result = gt.gen_tests(ADD_CODE)
    assert len(prompts) == 1
    assert "通过（第 1 次生成）" in result


@pytest.mark.parametrize("marker,expected", [
    ('@pytest.mark.skip(reason="暂未执行")', 3),
    ('@pytest.mark.xfail(reason="已知错误")', 4),
    ('@pytest.mark.xfail(reason="应当失败",strict=False)', 3),
])
@pytest.mark.parametrize("with_pass", [False,True])
def test_ignored_cases_are_not_all_passed(marker,expected,with_pass):
    #1.即使另有普通测试通过，skip、xfail和非严格xpass也不能冒充全部通过
    test = ("import pytest\nfrom target import add\n" + marker
            + f"\ndef test_ignored():\n    assert add(1,2) == {expected}\n")
    if with_pass:
        test += GOOD_TEST.replace("test_add", "test_passed")
    ok,output = gt.run_pytest(ADD_CODE,test)
    assert ok is False
    assert "没有全部实际通过" in output


def test_xfail_retries_even_with_full_coverage(monkeypatch):
    #1.预期失败也会执行被测代码，100%覆盖率不能替代真正通过
    test = "import pytest\n@pytest.mark.xfail(reason='known defect')\n" + BAD_TEST.replace("from target import add\n\n", "")
    test = "from target import add\n" + test
    prompts = fake_llm(monkeypatch,[test,GOOD_TEST])
    result = gt.gen_tests(ADD_CODE)
    assert "没有全部实际通过" in prompts[1][-1]["content"]
    assert "通过（第 2 次生成），覆盖率 100%" in result


@pytest.mark.parametrize("report", [None,"not json"])
def test_missing_or_invalid_pytest_report_is_not_passed(monkeypatch,report):
    from pathlib import Path

    #1.不能因进程返回0或打印passed就信任没有结构化结果的运行
    def run(*args,**kwargs):
        if report is not None:
            Path(kwargs["cwd"],"pytest_results.json").write_text(report,encoding="utf-8")
        return SimpleNamespace(returncode=0,stdout="1 passed",stderr="")

    monkeypatch.setattr(gt.subprocess,"run",run)
    ok,output = gt.run_pytest(ADD_CODE,GOOD_TEST)
    assert ok is False and "pytest结果统计不可用" in output
