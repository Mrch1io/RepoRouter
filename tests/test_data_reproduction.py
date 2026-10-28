"""The offline replay must fail before writing when evidence is inconsistent."""
import json
from importlib import import_module
import runpy
import sqlite3
import sys

import pytest

from experiments.routing_data import reproduce_data as replay
from experiments.routing_data import apply_review
from lora import format_to_sharegpt as sharegpt
from src.evaluation import e2e_manifest


def write_sources(tmp_path, monkeypatch, *, duplicate_id=False):
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(replay, "RAW", raw)
    files = (
        ("ft_qa_500_train.reviewed.jsonl", 497, "v1"),
        ("ft_qa_100_implicit_search.jsonl", 100, "implicit"),
        ("ft_qa_40_concept_no_tool.jsonl", 40, "concept"),
    )
    combined = []
    for name, count, prefix in files:
        rows = [{"sample_id": f"{prefix}_{i}"} for i in range(count)]
        if duplicate_id and prefix == "implicit":
            rows[0]["sample_id"] = "v1_0"
        combined.extend(rows)
        (raw / name).write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    return raw, combined


def test_existing_output_is_preserved(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "routing_train.json"
    sentinel.write_text("original evidence", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        replay.reproduce(destination)
    assert sentinel.read_text(encoding="utf-8") == "original evidence"


def test_duplicate_ids_write_nothing(tmp_path, monkeypatch):
    write_sources(tmp_path, monkeypatch, duplicate_id=True)
    destination = tmp_path / "result"
    with pytest.raises(ValueError, match="duplicate sample_id"):
        replay.reproduce(destination)
    assert not destination.exists()


def test_changed_snapshot_writes_nothing(tmp_path, monkeypatch):
    raw, rows = write_sources(tmp_path, monkeypatch)
    rows[0] = {"sample_id": "changed_after_training"}
    (raw / "ft_qa_train_v2.jsonl").write_text(
        "\n".join(map(json.dumps, rows)), encoding="utf-8"
    )
    destination = tmp_path / "result"
    with pytest.raises(ValueError, match="differ from the saved v2"):
        replay.reproduce(destination)
    assert not destination.exists()


def conversion_input(tmp_path):
    #1.只在临时目录造输入，不使用历史训练文件
    path = tmp_path / "input.json"
    rows = [{"query": "什么是列表", "decision_type": "no_tool"},
            {"query": "什么是字典", "decision_type": "no_tool"}]
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path, rows


def run_conversion(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["format_to_sharegpt", *map(str, args)])
    sharegpt._main()


def test_split_without_eval_output_preserves_training(tmp_path, monkeypatch):
    source, _ = conversion_input(tmp_path)
    target = tmp_path / "train.json"
    target.write_text("旧训练数据", encoding="utf-8")
    with pytest.raises(SystemExit):
        run_conversion(monkeypatch, "--input", source, "--out-train", target, "--split", "0.5")
    assert target.read_text(encoding="utf-8") == "旧训练数据"


@pytest.mark.parametrize("existing", ["train", "eval"])
def test_conversion_refuses_existing_output(tmp_path, monkeypatch, existing):
    source, _ = conversion_input(tmp_path)
    train, evaluation = tmp_path / "train.json", tmp_path / "eval.json"
    target = train if existing == "train" else evaluation
    other = evaluation if existing == "train" else train
    target.write_text("旧数据", encoding="utf-8")
    with pytest.raises(SystemExit):
        run_conversion(monkeypatch, "--input", source, "--out-train", train,
                       "--split", "0.5", "--out-eval", evaluation)
    assert target.read_text(encoding="utf-8") == "旧数据"
    assert not other.exists()


@pytest.mark.parametrize("collision", ["outputs", "input"])
def test_conversion_rejects_path_collision(tmp_path, monkeypatch, collision):
    source, _ = conversion_input(tmp_path)
    original = source.read_bytes()
    train = tmp_path / "train.json"
    evaluation = train if collision == "outputs" else source
    with pytest.raises(SystemExit):
        run_conversion(monkeypatch, "--input", source, "--out-train", train,
                       "--split", "0.5", "--out-eval", evaluation)
    assert source.read_bytes() == original
    assert not train.exists()


def test_invalid_eval_writes_no_training(tmp_path, monkeypatch):
    source, _ = conversion_input(tmp_path)
    bad_eval = tmp_path / "bad_eval.json"
    bad_eval.write_text('[{"decision_type":"unknown"}]', encoding="utf-8")
    destination = tmp_path / "result"
    with pytest.raises(ValueError, match="decision_type"):
        run_conversion(monkeypatch, "--input", source, "--out-train", destination / "train.json",
                       "--eval-input", bad_eval, "--out-eval", destination / "eval.json")
    assert not destination.exists()


def test_conversion_writes_new_train_and_eval(tmp_path, monkeypatch):
    source, rows = conversion_input(tmp_path)
    train, evaluation = tmp_path / "train.json", tmp_path / "eval.json"
    run_conversion(monkeypatch, "--input", source, "--out-train", train,
                   "--eval-input", source, "--out-eval", evaluation)
    expected = sharegpt.convert_to_sharegpt(rows)
    assert json.loads(train.read_text(encoding="utf-8")) == expected
    assert json.loads(evaluation.read_text(encoding="utf-8")) == expected


def test_cross_repo_refuses_existing_output(tmp_path, monkeypatch):
    target = tmp_path / "old_eval.json"
    target.write_text("旧评测题", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["build_cross_repo_eval", "--out", str(target)])
    with pytest.raises(SystemExit):
        runpy.run_module("lora.build_cross_repo_eval", run_name="__main__")
    assert target.read_text(encoding="utf-8") == "旧评测题"


@pytest.mark.parametrize("no_summary", [False, True])
def test_cross_repo_writes_new_output(tmp_path, monkeypatch, no_summary):
    #1.题集与目标都换成测试数据，避免重写历史87题
    monkeypatch.setattr(e2e_manifest, "build_e2e_manifest", lambda: [
        {"id": "a", "bucket_type": "domain", "query": "查函数", "should_call_tool": "yes"},
        {"id": "b", "bucket_type": "domain", "query": "概念", "should_call_tool": "no"},
        {"id": "c", "bucket_type": "domain", "query": "可选", "should_call_tool": "optional"},
    ])
    target = tmp_path / "new" / "eval.json"
    args = ["build_cross_repo_eval", "--out", str(target)]
    if no_summary:
        args.append("--no-repo-summary")
    monkeypatch.setattr(sys, "argv", args)
    runpy.run_module("lora.build_cross_repo_eval", run_name="__main__")
    rows = json.loads(target.read_text(encoding="utf-8"))
    assert [row["id"] for row in rows] == ["a", "b"]
    expected = sharegpt.routing_system_prompt(None) if no_summary else sharegpt.load_prompt_snapshot()
    assert all(row["conversations"][0]["value"] == expected for row in rows)
    assert json.loads(rows[0]["conversations"][2]["value"])["tool_name"] == "search_codebase"


def test_review_refuses_existing_output(tmp_path, monkeypatch):
    target = tmp_path / "reviewed.jsonl"
    target.write_text("已复核数据", encoding="utf-8")
    monkeypatch.setattr(apply_review, "OUT", target)
    monkeypatch.setattr(apply_review, "INP", tmp_path / "missing.jsonl")
    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        apply_review.main()
    assert target.read_text(encoding="utf-8") == "已复核数据"


def test_review_writes_new_output(tmp_path, monkeypatch):
    source, target = tmp_path / "input.jsonl", tmp_path / "reviewed.jsonl"
    ids = [*apply_review.TO_SEARCH, *apply_review.TO_ANALYSE_CODE, *apply_review.TO_NO_TOOL]
    rows = [{"sample_id": sid, "query": sid, "decision_type": "clarify_needed"} for sid in ids]
    source.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    original = source.read_bytes()
    monkeypatch.setattr(apply_review, "INP", source)
    monkeypatch.setattr(apply_review, "OUT", target)
    apply_review.main()
    result = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(result) == len(ids)
    assert result[0]["arguments"]["query"] == apply_review.TO_SEARCH[ids[0]]
    assert result[-1]["decision_type"] == "no_tool"
    assert source.read_bytes() == original


@pytest.mark.parametrize("state",["existing","concurrent","new"])
def test_retrieval_eval_preserves_output(tmp_path,monkeypatch,state):
    from langchain_core.documents import Document
    from src.config import settings
    from src.evaluation import rag_dataset
    from src.rag import vectorstore

    #1.只用一条假检索结果，已有文件必须在加载向量库前拒绝
    target = tmp_path / "result" / "retrieval.json"
    original = b"historical result\r\n"
    if state == "existing":
        target.parent.mkdir()
        target.write_bytes(original)
    calls = []

    class Store:
        def __init__(self,**kwargs):
            if state == "existing":
                pytest.fail("已有输出不应连接向量库")
            calls.append("init")

        def load(self):
            #2.模拟运行途中另一个进程抢先写入，最终保存也不能覆盖
            if state == "concurrent":
                target.parent.mkdir()
                target.write_bytes(original)

        def search_dense(self,query,top_k):
            calls.append(query)
            return [Document(page_content="def f(): pass",metadata={"source":"f.py"})]

        search_bm25 = search_dense
        search_hybrid = search_dense

    monkeypatch.setattr(vectorstore,"VectorStore",Store)
    monkeypatch.setattr(rag_dataset,"EVAL_QUERIES",[
        {"query":"f","expected_files":["f.py"],"query_type":"test"},
    ])
    for name in ("Chroma_persist_dir","bm25_persist_dir","qwen_embedding_name"):
        monkeypatch.setattr(settings,name,getattr(settings,name))
    monkeypatch.setattr(sys,"argv",["eval_retrieval","--json-out",str(target)])

    if state == "existing":
        with pytest.raises(SystemExit) as error:
            runpy.run_module("src.evaluation.eval_retrieval",run_name="__main__")
        assert error.value.code == 2 and calls == []
    elif state == "concurrent":
        with pytest.raises(FileExistsError):
            runpy.run_module("src.evaluation.eval_retrieval",run_name="__main__")
    else:
        runpy.run_module("src.evaluation.eval_retrieval",run_name="__main__")
    if state != "new":
        assert target.read_bytes() == original
    else:
        result = json.loads(target.read_text(encoding="utf-8"))
        assert result["n"] == 1 and set(result["modes"]) == {"dense","bm25","hybrid"}
        assert all(row["rr"] == [1.0] for row in result["modes"].values())
        assert calls == ["init","f","f","f"]


def test_retrieval_eval_rejects_dangling_symlink(tmp_path,monkeypatch):
    from src.rag import vectorstore

    #1.悬空链接的exists为假，也不能当成空闲输出路径
    target = tmp_path / "retrieval.json"
    missing = tmp_path / "missing.json"
    try:
        target.symlink_to(missing)
    except OSError:
        pytest.skip("当前系统不允许创建符号链接")
    monkeypatch.setattr(vectorstore,"VectorStore",lambda **kw: pytest.fail("悬空链接应在加载前拒绝"))
    monkeypatch.setattr(sys,"argv",["eval_retrieval","--json-out",str(target)])
    with pytest.raises(SystemExit) as error:
        runpy.run_module("src.evaluation.eval_retrieval",run_name="__main__")
    assert error.value.code == 2
    assert target.is_symlink() and not missing.exists()


@pytest.mark.parametrize("script,flag",[("update_grounding_labels","--out"),("eval_soft_fuse","--json-out")])
@pytest.mark.parametrize("state",["file","directory","symlink"])
def test_diagnostic_output_rejected_before_loading(tmp_path,monkeypatch,script,flag,state):
    #1.目录和悬空链接也不能当成可写的新结果；拒绝时不能读库或调用模型
    module = import_module(f"scripts.{script}")
    target,missing = tmp_path / "result.json",tmp_path / "missing.json"
    original = b"historical result\r\n"
    if state == "file":
        target.write_bytes(original)
    elif state == "directory":
        target.mkdir()
        (target / "keep.json").write_bytes(original)
    else:
        try:
            target.symlink_to(missing)
        except OSError:
            pytest.skip("当前系统不允许创建符号链接")
    monkeypatch.setattr(module,"ROOT",tmp_path)
    argv = [script,flag,str(target)]
    if script == "update_grounding_labels":
        db = tmp_path / "unused.sqlite3"
        db.touch()
        argv += ["--sqlite",str(db)]
        monkeypatch.setattr(module,"load_index_text",lambda path: pytest.fail("已有输出不应读取数据库"))
    else:
        for name in ("Chroma_persist_dir","bm25_persist_dir","qwen_embedding_name"):
            monkeypatch.setattr(module.settings,name,getattr(module.settings,name))
        monkeypatch.setattr(module,"VectorStore",lambda: pytest.fail("已有输出不应初始化向量库"))
    monkeypatch.setattr(sys,"argv",argv)
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    if state == "symlink":
        assert target.is_symlink() and not missing.exists()
    else:
        saved = target / "keep.json" if state == "directory" else target
        assert saved.read_bytes() == original


def test_grounding_labels_require_explicit_output(tmp_path,monkeypatch):
    from scripts import update_grounding_labels as module

    #1.不指定候选路径就退出，即使旧默认文件尚不存在也不能自动启用新标签
    db = tmp_path / "unused.sqlite3"
    db.touch()
    monkeypatch.setattr(module,"ROOT",tmp_path)
    monkeypatch.setattr(module,"load_index_text",lambda path: pytest.fail("缺少--out时不应读取数据库"))
    monkeypatch.setattr(sys,"argv",["update_grounding_labels","--sqlite",str(db)])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("state",["concurrent","new"])
def test_grounding_labels_preserve_evidence(tmp_path,monkeypatch,state):
    from scripts import update_grounding_labels as module

    #1.真实小库和源码产生两种降级理由，候选与当前评分依据分开保存
    db,source = tmp_path / "chroma.sqlite3",tmp_path / "src" / "memory.py"
    source.parent.mkdir()
    source.write_text("def _prune_memory(): pass\n",encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    conn = sqlite3.connect(db)
    try:
        conn.executescript("CREATE TABLE embeddings (id INTEGER); INSERT INTO embeddings VALUES (1);"
                           "CREATE TABLE embedding_metadata (key TEXT, string_value TEXT);"
                           "INSERT INTO embedding_metadata VALUES ('chroma:document','def known(): pass');")
        conn.commit()
    finally:
        conn.close()
    answers,judges = tmp_path / "answers.jsonl",tmp_path / "judges.jsonl"
    questions = {"p12":"_prune_memory 怎样裁剪记忆？","p19":"scripts目录有哪些文件？"}
    answers.write_text("\n".join(json.dumps({"sample_id":sid,"question":q},ensure_ascii=False)
                                for sid,q in questions.items()),encoding="utf-8")
    judges.write_text("\n".join(json.dumps({"sample_id":sid,"final_judgement":"incorrect"})
                               for sid in questions),encoding="utf-8")
    active = tmp_path / "data" / "chroma_verified_skip_must_ground.json"
    active.parent.mkdir()
    active.write_text(json.dumps({"entries":{"p12":{"tier":"must_ground"}}}),encoding="utf-8")
    originals = {path:path.read_bytes() for path in (db,source,answers,judges,active)}
    monkeypatch.setattr(module,"ROOT",tmp_path)
    monkeypatch.setattr(e2e_manifest,"_CHROMA_VERIFIED_JSON",active)
    before = e2e_manifest._index_aware_overrides_merged()
    target = tmp_path / "candidate" / "labels.json"
    occupied = b"concurrent historical result\r\n"
    read_index = module.load_index_text

    def load_index(path):
        text = read_index(path)
        if state == "concurrent":
            target.parent.mkdir()
            target.write_bytes(occupied)
        return text

    monkeypatch.setattr(module,"load_index_text",load_index)
    monkeypatch.setattr(sys,"argv",["update_grounding_labels","--sqlite",str(db),"--judge",str(judges),
                                    "--e2e",str(answers),"--src-root",str(source.parent),"--out",str(target)])
    #2.预检之后被抢占也不能覆盖；新路径必须保留真实诊断内容
    if state == "concurrent":
        with pytest.raises(FileExistsError):
            module.main()
        assert target.read_bytes() == occupied
    else:
        module.main()
        result = json.loads(target.read_text(encoding="utf-8"))
        assert result["embeddings_count"] == 1 and result["chroma_document_chars"] == len("def known(): pass")
        assert set(result["entries"]) == {"p12","p19"}
        assert result["entries"]["p12"]["tier"] == "prefer_ground"
        assert result["entries"]["p12"]["reason"] == "src_yes_chroma_no"
        assert result["entries"]["p12"]["missing_tokens_in_chroma"] == ["_prune_memory"]
        assert result["entries"]["p19"]["tier"] == "ok_generic"
        assert result["entries"]["p19"]["reason"] == "scripts_scope"
    assert all(path.read_bytes() == content for path,content in originals.items())
    assert e2e_manifest._index_aware_overrides_merged() == before


@pytest.mark.parametrize("state",["concurrent","new","no_output"])
def test_soft_fuse_preserves_output(tmp_path,monkeypatch,capsys,state):
    from langchain_core.documents import Document
    from scripts import eval_soft_fuse as module

    #1.只替换检索边界，距离收集、AUC和阈值选择仍走真实计算
    distances = {"domain":1.0,"far_general":1.8,"term_near_no_rag":0.8,"misleading_edge":0.6}
    names = ("DOMAIN_IN_PROJECT","OUTDOMAIN_FAR_GENERAL","OUTDOMAIN_TERM_NEAR_NO_RAG","OUTDOMAIN_MISLEADING_EDGE")
    for name,split in zip(names,distances):
        monkeypatch.setattr(module,name,[{"id":split,"query":split}])
    for name,value in (("Chroma_persist_dir",str(tmp_path / "index")),
                       ("bm25_persist_dir",str(tmp_path / "index" / "bm25.pkl")),
                       ("qwen_embedding_name","test-embedding")):
        monkeypatch.setattr(module.settings,name,value)
    monkeypatch.setattr(module,"ROOT",tmp_path)
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "result" / "fuse.json"
    occupied = b"concurrent historical result\r\n"
    calls = []

    class Store:
        def __init__(self):
            calls.append("init")
        def load(self):
            calls.append("load")
            if state == "concurrent":
                target.parent.mkdir()
                target.write_bytes(occupied)
        def dense_top1_with_distance(self,query):
            calls.append(query)
            return Document(page_content=query,metadata={"source":f"{query}.py"}),distances[query]

    monkeypatch.setattr(module,"VectorStore",Store)
    argv = ["eval_soft_fuse"] + ([] if state == "no_output" else ["--json-out",str(target)])
    monkeypatch.setattr(sys,"argv",argv)
    if state == "concurrent":
        with pytest.raises(FileExistsError):
            module.main()
        assert target.read_bytes() == occupied
    else:
        module.main()
        printed = capsys.readouterr().out
        assert "ROC-AUC" in printed and "0.333" in printed and "1.0000" in printed
        if state == "no_output":
            assert "已写入" not in printed and list(tmp_path.iterdir()) == []
        else:
            result = json.loads(target.read_text(encoding="utf-8"))
            assert result["embedding"] == "test-embedding" and result["roc_auc"] == pytest.approx(1/3)
            assert result["recommended"] == {"threshold":1.0,"outdomain_interception_rate":1/3,
                                             "private_false_block_rate":0.0,"outdomain_intercepted":1,
                                             "private_intercepted":0}
            assert result["rows"] == [{"split":split,"id":split,"query":split,"top1_distance":distance,
                                       "top1_source":f"{split}.py"} for split,distance in distances.items()]
    assert calls == ["init","load",*distances]
