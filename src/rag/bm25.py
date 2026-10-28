from __future__ import annotations #必须在最前面
#bm25稀疏向量库
#  词袋模型：正则表达式过滤，处理代码精细搜索

import pickle #序列存储和读写
from heapq import nlargest

from typing import List
from pathlib import Path

import re   #正则表达式和模式匹配

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from src.config import settings


#1.分词：代码标识符保留整词，中文按单字切
#私有内部辅助函数，私有变量：标注_
def _tokenize(text : str)->List[str]:
    #原来的正则会把中文整句丢掉，现在保留标识符并把汉字按单字切分
    #r 表示原生字符串，不用再转义反斜杠
    #+匹配连续的标识符，|右边匹配单个汉字
    tokens = re.findall(r"[A-Za-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]",text)

    #返回切好的字符串列表(去空)，工整
    return [t for t in tokens if t]

class BM25Retriever:
    """基于BM25的稀疏检索器
    带文件存储和读写，search功能"""
    def __init__(self):
                    #把配置里面的str地址转成Path地址方便处理
                    #Chroma可以直接接受字符串去处理，但是自己搭建的bm25需要转成Path类方便遍历
        self.persist_path = Path(settings.bm25_persist_dir)
        self.docs : List[Document] = []
        #可能是None或者BM250kapi类
        self.bm25 : BM25Okapi | None = None
        self._initialized = False

    def _build_index(self):
        #先按每块的内容分词，列表表示式收集起来
        corpus_tokens = [_tokenize(doc.page_content) for doc in self.docs]
        #再计算稀疏词频矩阵，空语料不构造 BM25
        self.bm25 = BM25Okapi(corpus_tokens) if any(corpus_tokens) else None
        self._initialized = True



#2.写库存储:传入文件列表
    def index_document(self,docs:List[Document]):
        self.docs = docs
        print("正在计算bm25稀疏词频矩阵")
        self._build_index()

        #在上一层目录创建相应文件夹
        self.persist_path.parent.mkdir(parents=True,exist_ok=True) #parents = True:上一层文件夹都建立好，exist_ok:如果存在就静默
        with open(self.persist_path,"wb") as f:  #"wb"：写入二进制模式
            #存入文本和稀疏索引（这里不如Chroma智能，需要手动传入字典）
            pickle.dump({"docs" : self.docs,"bm25": self.bm25,"tokenizer_version":2},f)
        print(f"稀疏索引已持久化保存至{self.persist_path}")

#3.load加载数据库
    def load(self):
        #先看路径在不在
        if not self.persist_path.exists():

            #工程：使用raise RuntimeError报错终止！
            raise RuntimeError(f"bm25索引未建库！\n请先建库于{self.persist_path}")
        
        
        
        
        
        with open(self.persist_path,"rb") as f:#读取二进制模式
            #使用pikcle存储的序列需要先用pickle.load读取
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.docs = data["docs"]
        if data.get("tokenizer_version") != 2:
            #旧索引只在内存里重算，不覆盖硬盘文件
            print("[BM25] 索引分词版本缺失或不匹配，将在内存中按新版中文分词重算；历史检索评测数字可能变化，磁盘索引保持原样。")
            self._build_index()
        self._initialized = True
        print("成功从硬盘持久路径读取到bm25索引引擎！")




#4.search根据索引查找 传入查询和top_k
    def search(self,query:str,top_k = 5)->List[Document]:
        #先看bm25数据库有没有初始化
        if not self._initialized:
            raise RuntimeError(f"bm25数据库未初始化！\n请先调用index_document或者load函数")
        if self.bm25 is None or top_k <= 0:
            return []

        #将query也暴力切分
        query = _tokenize(query)
        if not query:
            return []
        #计算和bm25中各个文档的匹配分数
        scores = self.bm25.get_scores(query)

        #只给有共同词的文档排名，避免无关文档混入
        matched = (i for i,freqs in enumerate(self.bm25.doc_freqs) if any(t in freqs for t in query))
        #nlargest 按分数从大到小取前 k 项，不用给所有文档排序
        sorted_indices = nlargest(top_k,matched,key=lambda i:scores[i])
        #按照新的序号顺序返回文档列表
        return [self.docs[i] for i in sorted_indices]
