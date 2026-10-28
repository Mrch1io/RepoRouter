import pytest

from src.agent.tools.code_analyser import analyse_code
test_code = """import os
import sys
import json

def process_data(items=[]):
    result = []
    for item in items:
        if item > 0:
            if item % 2 == 0:
                try:
                    result.append(item * 2)
                except:
                    pass
        else:
            result.append(0)
    return result

def helper():
    pass"""


def test_analyse_code_finds_issues():
    report = analyse_code(test_code)
    print(report)
    #可变默认参数、裸except、没用到的import都应该被找出来
    assert "共发现5个问题" in report
    assert "默认可变参数" in report
    assert "裸except" in report
    assert "'os' 被导入但未使用" in report


def test_syntax_error():
    assert "语法错误" in analyse_code("def f(:\n    pass")


@pytest.mark.parametrize("source,line", [
    ("return 42",1),
    ("break",1),
    ("def f(x,x):\n    return x\n",1),
])
def test_compile_rejects_invalid_ast(source,line):
    #1.能解析成AST也不一定能编译，必须报告原来的出错行号
    report = analyse_code(source)
    assert "语法错误" in report and f"第{line}行" in report


def test_compile_check_does_not_execute_code(tmp_path):
    #1.编译只检查合法性，不能运行用户代码或产生文件
    path = tmp_path / "must_not_exist.txt"
    source = f"open({str(path)!r},'w').write('executed')"
    assert analyse_code(source) == "代码检查完成 未发现问题"
    assert not path.exists()
