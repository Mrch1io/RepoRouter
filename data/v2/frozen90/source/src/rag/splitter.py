#代码切分器：调用langchain工业级解决方案
from typing import List
#要用到langchain数据封装类
from langchain_core.documents import Document
from langchain_text_splitters import Language,RecursiveCharacterTextSplitter



class CodeSplitter:     #指定默认值
    def __init__(self,code_chunk_size:int = 2000,code_chunk_overlap:int = 350, #修改之后要重新建库
                 text_chunk_size:int = 1500,text_chunk_overlap:int = 350,mode:str = "char"):

        #mode = "char"：按字符数切（原来的做法）；"ast"：按语法树切，函数、类不会被切成两半
        #ast 借鉴 cAST 论文（yilinjz/astchunk）：大节点往下拆，相邻小节点往上合并
        self.mode = mode
        if mode == "ast":
            from astchunk import ASTChunkBuilder
            #astchunk 按“非空白字符数”算大小：1500 和字符切法的 2000（含缩进空白）大致相当，对比才公平
            self.ast_builder = ASTChunkBuilder(max_chunk_size=1500,language="python",metadata_template="default")

#1.初始化，构建专门的py代码分割。和一般的text分割           #指定from_language
        self.py_splitter = RecursiveCharacterTextSplitter.from_language(
            #2.规定超参数，进行分割
        language = Language.PYTHON,
        chunk_overlap = code_chunk_overlap,
        chunk_size = code_chunk_size


    )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_overlap = text_chunk_overlap,
            chunk_size = text_chunk_size
        )
        
    #3.构建最后的列表，分割的结果一一放进去
    #分割函数：接受文档列表，返回切分好块的列表
    #docs : List[Document] docs是传入的数据，List[Document]注明类型
    def split(self, docs: List[Document], *, show_progress: bool = False) -> List[Document]:
        doc_iter = docs
        if show_progress:
            try:
                from src.rag.tqdm_util import tqdm

                doc_iter = tqdm(docs, desc="切分 chunk", unit="file")
            except ImportError:
                pass
        all_chunks = []
        for doc in doc_iter:
            suffix = doc.metadata.get("suffix","")  #使用.get查询字典值：比较温和，否则如果查不到就会报错
            if suffix == ".py" and self.mode == "ast":
                chunk = self._split_ast(doc)
            elif suffix == ".py":
                            #py_splitter是对象：not callable不可调用，需要调用方法
                chunk = self.py_splitter.split_documents([doc])   #切分引擎接受列表
            else:
                chunk = self.text_splitter.split_documents([doc])
            #最后填入最终列表：.append()和.extend()的差别：前者整块放入，后者先解包再放入
            all_chunks.extend(chunk)

        return all_chunks

    #按语法树切一个 py 文件，顺便记下每块的起止行号（检索结果可以带行号）
    def _split_ast(self,doc:Document)->List[Document]:
        out = []
        for c in self.ast_builder.chunkify(doc.page_content):
            #空文件（比如 __init__.py）会切出空块：跳过，否则 embedding 接口会返回维度不一样的向量
            if not c["content"].strip():
                continue
            meta = dict(doc.metadata)
            meta["start_line"] = c["metadata"]["start_line_no"] + 1   #astchunk 从0开始，改成从1开始
            meta["end_line"] = c["metadata"]["end_line_no"] + 1
            out.append(Document(page_content=c["content"],metadata=meta))
        return out




