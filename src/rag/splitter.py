#代码切分器：调用langchain工业级解决方案
from typing import List
import logging
import re
#要用到langchain数据封装类
from langchain_core.documents import Document
from langchain_text_splitters import Language,RecursiveCharacterTextSplitter



class CodeSplitter:     #指定默认值
    def __init__(self,code_chunk_size:int = 2000,code_chunk_overlap:int = 350, #修改之后要重新建库
                 text_chunk_size:int = 1500,text_chunk_overlap:int = 350,mode:str = "char"):

        #mode = "char"：按字符数切（原来的做法）；"ast"：优先保持函数、类完整，大节点拆分后补上所属上下文
        #ast 借鉴 cAST 论文（yilinjz/astchunk）：大节点往下拆，相邻小节点往上合并
        self.mode = mode
        if mode == "ast":
            from astchunk import ASTChunkBuilder
            #保留历史实验的1500阈值；本地astchunk按UTF-8字节计数并忽略ASCII空白，不是字符数硬上限
            #code_chunk_size/overlap控制char及正文不完整时的回退；正常AST不加相邻节点重叠，祖先上下文另补
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
            #每个文件内部编号，区分内容相同的不同代码块
            for index, part in enumerate(chunk):
                part.metadata["chunk_id"] = index
            #最后填入最终列表：.append()和.extend()的差别：前者整块放入，后者先解包再放入
            all_chunks.extend(chunk)

        return all_chunks

    #按语法树切一个 py 文件，顺便记下每块的起止行号（检索结果可以带行号）
    def _split_ast(self,doc:Document)->List[Document]:
        out = []
        bodies = []
        #1.复用astchunk的祖先信息，不再解析一次源码
        for c in self.ast_builder.chunkify(doc.page_content,chunk_expansion=True):
            #行数只算原始正文；用它分开库补的头部，源码里有三引号也不受影响
            lines = c["content"].split("\n")
            body_start = len(lines) - c["metadata"]["line_count"]
            content = "\n".join(lines[body_start:])
            #空文件（比如 __init__.py）会切出空块：跳过，否则 embedding 接口会返回维度不一样的向量
            if not content.strip():
                continue
            bodies.append(content)
            #2.只保留类名、函数名，不把长参数列表重复塞进每块
            context = " > ".join(re.findall(r"^\s*((?:async\s+)?(?:class|def)\s+\w+)",
                                           "\n".join(lines[1:body_start-1]),re.MULTILINE))
            meta = dict(doc.metadata)
            meta["start_line"] = c["metadata"]["start_line_no"] + 1   #astchunk 从0开始，改成从1开始
            meta["end_line"] = c["metadata"]["end_line_no"] + 1
            if context:
                meta["context"] = context
                #补充文字参与检索，行号仍指向下面的原始正文
                content = f"#块起点上下文（非原始代码）：{context}\n{content}"
            out.append(Document(page_content=content,metadata=meta))
        #3.astchunk可能丢掉超长叶节点，核对正文后回退；不把库补的上下文算成源码
        if re.sub(r"[ \t\r\n\f\v]+","",doc.page_content) != re.sub(r"[ \t\r\n\f\v]+","","".join(bodies)):
            logging.warning("AST切分正文不完整，改用字符切分：%s",doc.metadata.get("source","未知文件"))
            return self.py_splitter.split_documents([doc])
        return out
