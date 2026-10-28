import logging
from pathlib import Path

import pytest

from src.rag.loader import RePoLoader


def test_loader_filters(tmp_path,caplog):
    #造一个小仓库：支持的后缀要读进来，忽略的目录要跳过
    (tmp_path / "a.py").write_text("print(1)",encoding="utf-8")
    (tmp_path / "B.java").write_text("class B {}",encoding="utf-8")
    (tmp_path / "pic.png").write_bytes(b"\x89PNG")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "c.py").write_text("x",encoding="utf-8")
    (tmp_path / "rag_dataset.py").write_text("EVAL_QUERIES = []",encoding="utf-8")   #考题文件不进索引

    docs = RePoLoader(tmp_path).load()
    names = sorted(d.metadata["file_name"] for d in docs)
    assert names == ["B.java","a.py"]   #.java 之前因为少了点读不进来
    assert docs[0].metadata["suffix"] in (".py",".java")
    assert not caplog.records


def test_loader_excludes_evidence_inside_root(tmp_path):
    #仓库里面的评测结果仍然要排除，普通源码不受影响
    for prefix in RePoLoader.EVIDENCE_DIRS:
        directory = tmp_path.joinpath(*prefix)
        directory.mkdir(parents=True)
        (directory / "result.py").write_text("answer = 1",encoding="utf-8")
    (tmp_path / "routing_train.json").write_text("{}",encoding="utf-8")
    (tmp_path / "engine.py").write_text("value = 1",encoding="utf-8")
    assert [d.metadata["source"] for d in RePoLoader(tmp_path).load()] == ["engine.py"]


def test_loader_reads_explicit_source_below_evidence(tmp_path):
    #显式选择冻结源码时，只过滤这个根目录里面的实验数据
    source = tmp_path / "data" / "v2" / "frozen90" / "source" / "src"
    source.mkdir(parents=True)
    (source / "engine.py").write_text("value = 1",encoding="utf-8")
    evidence = source / "data" / "v2"
    evidence.mkdir(parents=True)
    (evidence / "result.py").write_text("answer = 1",encoding="utf-8")
    assert RePoLoader(tmp_path).load() == []
    docs = RePoLoader(source).load()
    assert [(d.metadata["source"],d.page_content) for d in docs] == [("engine.py","value = 1")]


@pytest.mark.parametrize("error_type",[PermissionError,FileNotFoundError])
def test_loader_reports_read_error_and_keeps_other_files(tmp_path,monkeypatch,caplog,error_type):
    #1.只模拟读取边界的失败，其他文件仍从磁盘读取
    bad = tmp_path / "bad.py"
    good = tmp_path / "good.py"
    bad.write_text("private = 1",encoding="utf-8")
    good.write_text("value = 2",encoding="utf-8")
    read_text = Path.read_text
    reason = "模拟文件读取失败"

    def read(path,*args,**kwargs):
        if path == bad:
            raise error_type(reason)
        return read_text(path,*args,**kwargs)

    monkeypatch.setattr(Path,"read_text",read)
    docs = RePoLoader(tmp_path).load()
    assert [(doc.page_content,doc.metadata) for doc in docs] == [
        ("value = 2",{"source":"good.py","file_name":"good.py","suffix":".py"})
    ]
    #2.失败路径和原因要能定位，不能只返回缺文件的结果
    warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert str(bad) in warnings[0].getMessage() and reason in warnings[0].getMessage()


def test_loader_does_not_swallow_unexpected_error(tmp_path,monkeypatch):
    #1.程序错误不能被当成普通文件读取失败跳过
    target = tmp_path / "code.py"
    target.write_text("value = 1",encoding="utf-8")
    read_text = Path.read_text
    error = RuntimeError("模拟程序错误")

    def read(path,*args,**kwargs):
        if path == target:
            raise error
        return read_text(path,*args,**kwargs)

    monkeypatch.setattr(Path,"read_text",read)
    with pytest.raises(RuntimeError) as caught:
        RePoLoader(tmp_path).load()
    assert caught.value is error


if __name__ == "__main__":
    loader = RePoLoader(".")#当前目录
    docs = loader.load()
    print("文件个数：",len(docs))
    #查看前5个文件情况
    for d in docs[:5]:
        print("来源:       ",d.metadata["source"])
        print("文件名称：     ",d.metadata["file_name"]) #键是字符串，需要带引号！
        print("内容前100个词：         ",d.page_content[:100])
