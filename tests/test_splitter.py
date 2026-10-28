import pytest
from langchain_core.documents import Document
from src.rag.splitter import CodeSplitter


def test_split_long_python_file():
    code = "\n\n".join(f"def f{i}():\n    return {i}\n" for i in range(400))
    doc = Document(page_content=code,metadata={"suffix":".py","source":"x.py"})
    chunks = CodeSplitter().split([doc])
    #切成多块，每块不超过 chunk_size，元数据保留
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 2000 for c in chunks)
    assert all(c.metadata["source"] == "x.py" for c in chunks)


def test_ast_preserves_functions():
    #按语法树切：这些短函数都完整地落在某一块里，不会被切成两半
    code = "\n\n".join(f"def f{i}(x):\n    y = x + {i}\n    return y * 2\n" for i in range(200))
    doc = Document(page_content=code,metadata={"suffix":".py","source":"x.py"})
    chunks = CodeSplitter(mode="ast").split([doc])
    assert len(chunks) > 1
    for i in range(200):
        assert any(f"def f{i}(x):\n    y = x + {i}\n    return y * 2" in c.page_content for c in chunks)
    #行号从1开始，和源文件对得上
    lines = code.splitlines()
    for c in chunks:
        assert c.page_content.splitlines()[0] == lines[c.metadata["start_line"] - 1]
        assert "context" not in c.metadata


@pytest.mark.parametrize("header,indent,context",[
    ("def process():\n","    ","def process"),
    ('class Worker:\n    async def process(\n        self,\n        value: str = """default""",\n    ):\n',
     "        ","class Worker > async def process"),
])
def test_ast_long_function_keeps_context_and_source_lines(header,indent,context):
    #大函数的后半块也带所属名称；多行参数、三引号不应混入补充的上下文
    code = header + "".join(f'{indent}value_{i} = "payload-{i}-abcdefghij"\n' for i in range(180))
    code += f"{indent}return value_0\n"
    doc = Document(page_content=code,metadata={"suffix":".py","source":"worker.py"})
    chunks = CodeSplitter(mode="ast").split([doc])
    assert len(chunks) > 2
    lines = code.split("\n")
    for index,c in enumerate(chunks):
        prefix,body = c.page_content.split("\n",1)
        assert prefix == f'#块起点上下文（非原始代码）：{c.metadata["context"]}'
        assert len(prefix) < 100
        #起止行号只对应原始正文，不把补的说明算成一行源码
        assert body == "\n".join(lines[c.metadata["start_line"]-1:c.metadata["end_line"]])
        assert c.metadata["source"] == "worker.py"
        assert c.metadata["chunk_id"] == index
        if c.metadata["start_line"] > header.count("\n"):
            assert c.metadata["context"] == context


@pytest.mark.parametrize("code",["","\n\n","   \n  "])
def test_ast_skips_empty_code(code):
    doc = Document(page_content=code,metadata={"suffix":".py"})
    assert CodeSplitter(mode="ast").split([doc]) == []


def test_character_parameters_do_not_change_intact_ast_chunks():
    #1.字符大小和重叠确实改变char分块，但不改变保留历史阈值的AST分块
    code = "\n\n".join(f"def f{i}(value):\n    return value + {i}\n" for i in range(120))
    doc = Document(page_content=code,metadata={"suffix":".py","source":"example.py"})
    small = dict(code_chunk_size=400,code_chunk_overlap=0)
    large = dict(code_chunk_size=2500,code_chunk_overlap=300)
    char_small = CodeSplitter(**small).split([doc])
    char_large = CodeSplitter(**large).split([doc])
    assert len(char_small) > len(char_large)
    assert all(len(chunk.page_content) <= 400 for chunk in char_small)
    ast_small = CodeSplitter(mode="ast",**small).split([doc])
    ast_large = CodeSplitter(mode="ast",**large).split([doc])
    assert ast_small == ast_large
    assert all("start_line" in chunk.metadata for chunk in ast_small)


@pytest.mark.parametrize("code",[
    'value = "' + "a" * 2000 + '"\n',
    'value = "' + "中" * 2000 + '"\n',
    'value = "' + "\u3000" * 2000 + '"\n',
    "#" + "long_comment" * 200 + "\n",
])
def test_ast_falls_back_when_long_leaf_content_is_lost(code,caplog):
    #1.正文不完整时沿用字符切法及其边界空白处理，不伪造AST行号
    doc = Document(page_content=code,metadata={"suffix":".py","source":"long.py"})
    options = dict(code_chunk_size=800,code_chunk_overlap=0)
    expected = CodeSplitter(**options).split([doc])
    chunks = CodeSplitter(mode="ast",**options).split([doc])
    assert chunks == expected and chunks
    #2.独立核对非空白负载，避免两边都只剩引号却通过比较
    content = "".join(chunk.page_content for chunk in chunks)
    for token in ("a","中","long_comment"):
        assert content.count(token) == code.count(token)
    assert "改用字符切分：long.py" in caplog.text
    assert all(chunk.metadata == {"suffix":".py","source":"long.py","chunk_id":index}
               for index,chunk in enumerate(chunks))
