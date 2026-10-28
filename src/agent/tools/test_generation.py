import importlib.util
import json
import os
import re
import subprocess  #开子进程跑pytest
import sys
import tempfile    #临时目录，跑完自动删掉
from pathlib import Path

from src.agent.llm import chat_completion

#借鉴 qodo-cover：测试光能通过还不够，覆盖率也要达标才算完成
MIN_COVERAGE = 80
#装了 pytest-cov 才统计覆盖率，没装就只看通过没通过
HAS_COV = importlib.util.find_spec("pytest_cov") is not None

TEST_GEN_SYSTEM = """你是一名资深 Python 测试工程师。
根据给定的函数代码和说明，生成 pytest 样式的单元测试。
要求：
1. 覆盖正常情况和常见边界情况
2. 使用 pytest 风格的测试函数 (test_xxx)
3. 必要时使用 fixture 或参数化
4. 测试代码要可运行，不要省略 import
5. 被测代码保存在 target.py 里，测试里用 from target import ... 导入
只输出测试代码，不要解释。
"""

#去掉模型可能带上的 ```python ``` 围栏
def _strip_fence(text : str)->str:
    text = re.sub(r"^\s*```(?:python)?\s*","",text or "")
    return re.sub(r"\s*```\s*$","",text)


#真的跑一遍pytest：返回(是否通过,输出)
def run_pytest(code : str,test_code : str,timeout : int = 30)->tuple:
    #1.被测代码和测试写进临时目录（with结束后目录自动删除）
    with tempfile.TemporaryDirectory() as d:
        Path(d,"target.py").write_text(code,encoding="utf-8")
        Path(d,"test_target.py").write_text(test_code,encoding="utf-8")
        #记录pytest自身的结果分类，不能把退出码0或覆盖率当成全部通过
        Path(d,"conftest.py").write_text("""import json
from pathlib import Path

def pytest_terminal_summary(terminalreporter):
    names = ("passed","failed","error","skipped","xfailed","xpassed","deselected")
    counts = {name: len(terminalreporter.stats.get(name, [])) for name in names}
    Path(__file__).with_name("pytest_results.json").write_text(json.dumps(counts),encoding="utf-8")
""",encoding="utf-8")

        #2.子进程里跑，设超时：生成的代码可能有死循环
        #term-missing：顺便列出没覆盖到的行号，喂回给模型补测试
        cmd = [sys.executable,"-m","pytest","-q","-p","no:cacheprovider","test_target.py"]
        if HAS_COV:
            cmd += ["--cov=target","--cov-report=term-missing"]
        try:
            r = subprocess.run(
                cmd,
                cwd=d,capture_output=True,text=True,encoding="utf-8",errors="replace",
                timeout=timeout,env={**os.environ,"PYTHONIOENCODING":"utf-8"},
            )
        except subprocess.TimeoutExpired:
            return False,f"pytest 运行超过 {timeout} 秒，已终止"

        #3.至少一条真正通过，而且没有跳过、预期失败或意外通过
        output = r.stdout + r.stderr
        if r.returncode != 0:
            return False,output
        try:
            counts = json.loads(Path(d,"pytest_results.json").read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError) as e:
            return False,output + f"\npytest结果统计不可用：{e}"
        ok = counts["passed"] > 0 and not any(n for name,n in counts.items() if name != "passed")
        if not ok:
            output += f"\n测试没有全部实际通过：{counts}"
        return ok,output


#从pytest-cov的表格里读出target.py的覆盖率，比如 “target.py   10   2   80%   5, 9” -> 80
def read_coverage(output : str)->int | None:
    m = re.search(r"target\.py\s+\d+\s+\d+\s+(\d+)%",output)
    return int(m.group(1)) if m else None


def gen_tests(code:str,description:str = "")->str:
    user_prompt = "下面是一段函数的代码，请为其生成单元测试"
    #之前这里写成了 f"\ncode\n"，代码根本没传进去
    user_prompt += f"\n```python\n{code}\n```\n"

    if description.strip():
        user_prompt += f"\n函数说明：{description}\n"

    message = [
        {"role":"system","content":TEST_GEN_SYSTEM},
        {"role":"user","content":user_prompt},
    ]
    test_code = _strip_fence(chat_completion(message,max_tokens=1500,temperature=0.2).content)

    #生成完真的跑一遍：失败或覆盖率不够，就带着报错/没覆盖的行让模型重写，最多重写2次
    for attempt in range(3):
        ok,output = run_pytest(code,test_code)
        cov = read_coverage(output)
        #装了pytest-cov却没有覆盖率，也按失败反馈给模型重写
        if HAS_COV and cov is None:
            ok = False
            output += "\n没有拿到 target.py 的覆盖率，测试可能没有导入被测代码（应该 from target import ...）"
        cov_text = f"，覆盖率 {cov}%" if cov is not None else ""
        #通过，并且覆盖率达标（没装 pytest-cov 时只看通过）
        if ok and (cov is None or cov >= MIN_COVERAGE):
            return f"{test_code}\n\n# pytest 结果：通过（第 {attempt + 1} 次生成）{cov_text}"
        if attempt == 2:
            break
        #把上一版测试和问题放回对话，让模型对着改
        if ok:
            feedback = f"测试都通过了，但覆盖率只有 {cov}%（目标 {MIN_COVERAGE}%）。下面 Missing 列是没覆盖到的行号，请补充测试，只输出完整测试代码：\n{output[-2000:]}"
        else:
            feedback = f"测试运行失败，pytest 输出如下。请修正测试代码（如果是被测代码本身的 bug，就在测试里用注释指出），只输出完整测试代码：\n{output[-2000:]}"
        message += [
            {"role":"assistant","content":test_code},
            {"role":"user","content":feedback},
        ]
        test_code = _strip_fence(chat_completion(message,max_tokens=1500,temperature=0.2).content)

    if ok:
        return f"{test_code}\n\n# pytest 结果：通过{cov_text}，但没达到 {MIN_COVERAGE}% 的目标"
    return f"{test_code}\n\n# pytest 结果：3 次都没通过，最后一次输出：\n{output[-1500:]}"
