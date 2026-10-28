#载入权威开源向量数据库模型pip install langchain-huggingface sentence-transformers


#阿里云SDK
import dashscope
import re
from http import HTTPStatus
from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma       #载入向量数据库:用于存储调用高维语义向量
#线上模型
from langchain_openai import OpenAIEmbeddings        #Embeddings词嵌入模型，转为高维向量

#线下模型（huggingface本地引擎、CrossEncoder精排）改成用到时再导入：
#不用本地模型的话，就不用装 torch 这些大依赖

#模型调用的配置文件
from src.config import settings
from src.rag.bm25 import BM25Retriever

class VectorStore:
    """
    使用chroma构建向量数据库引擎
    """
    #1.初始化，策略模型选择，构建好embedding模型，构建self._db = None


    #2.写入库：传入切分好的词块列表，需要的持久地址，嵌入模型传入Chroma.from_documents()自动建库


    #3.读库，服务器重启，直接.load()从原来的持久地址读取到self._db
    #建立向量数据库
    #也要指定embedding，因为要基于向量数据库把用户提问转化为向量


    #4.接受疑问和搜索

    #需要排错一下None的情况
    #接受疑问，设定top_K，会自动计算余弦相似度返回需要的值列表
                        #热插拔的rerank:默认为纯Dense.
    def __init__(self,use_reranker:bool = False):
        """混合检索算法"""
        #持久化地址
        self.persist_dir = settings.Chroma_persist_dir
        if settings.embedding_provider == "local":
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                #本地模型传入模型名即可
                model = settings.local_embedding_name
            )
        elif settings.embedding_provider == "qwen":
            self.embeddings = OpenAIEmbeddings(
                #区分聊天模型model_name  和embedding模型
                model = settings.qwen_embedding_name,
                openai_api_key = settings.qwen_api_key,
                openai_api_base = settings.qwen_base_url, 
                check_embedding_ctx_length=False,
                tiktoken_enabled=False,
                chunk_size=10,
                #显式指定维度：qwen3.7 的一批里只要有空字符串，整批会变成 2560 维，和库里的 1024 维对不上
                dimensions=1024,

            )
        else:
            self.embeddings = OpenAIEmbeddings(
                #区分聊天模型model_name  和embedding模型
                model = settings.embedding_model,
                openai_api_key = settings.deepseek_api_key,
                openai_api_base = settings.deepseek_base_url, 
                
            )
            #加下划线：关键私有变量：人为协定
        self._db = None
        self._repo_summary = None
        #bm25实例
        self.bm25_retriever = BM25Retriever()


        #独立初始化reranker模型
        self.use_reranker = use_reranker
        self.reranker = None

        self.reranker_name = ""

        if self.use_reranker:
            if settings.reranking_provider == "local":
                #明确要求重排时，模型加载失败就报错，不能改成普通检索继续计分
                from sentence_transformers import CrossEncoder #引入精排模型
                self.reranker_name = "BAAI/bge-reranker-v2-m3"   #多语言版，中文也能用
                self.reranker = CrossEncoder(self.reranker_name)
                print(f"本地reranker模型加载成功\n")
            #qwen rerank
            elif settings.reranking_provider == "qwen":
                #挂载云端api
                dashscope.api_key=settings.qwen_api_key
                self.reranker_name = settings.qwen_rerank_name
                if not dashscope.api_key:
                    raise ValueError("启用云端重排需要填写 qwen_api_key")
                else:
                    print("挂载云端qwen api成功")
            else:
                raise ValueError(f"不支持的重排方式：{settings.reranking_provider}")

    
    def index_document(self,all_chunks : List[Document]) -> None:

        self._repo_summary = None
        #构建向量数据库 .from_documents读库
        self._db = Chroma.from_documents(
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            documents=all_chunks,
        )
        #完成之后打印调试信息
        print(f"DenseChroma建库成功将{len(all_chunks)}个代码块转化为了向量,存储到路径{self.persist_dir}")

        #bm25建库
        self.bm25_retriever.index_document(all_chunks)
        print(f"bm25建库成功！储存到路径{settings.bm25_persist_dir}!")

    #从硬盘中读取数据库，不是返回，就是重新根据已有数据再建一个Chroma数据库实例
    def load(self)->None:

        self._repo_summary = None
        #使用raise处理错误：立刻终止运行，爆出错误情况
        #return就返回一个值，不易排错
        #必须使用is None判断：安全性。比较物理地址
        # 不需要判断数据库是否为空

        #因为在core中实例化的就是空的，但是是从原来setting中设定的恒久路径读取数据库，就只需要初始化一次！
        # if self._db is None:
        #     raise RuntimeError("向量数据库未初始化，读取失败")


        
        self._db = Chroma(
            #不再是embedding了，而是embeddingfunction，作用于查询
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
            )
        print(f"成功从路径{self.persist_dir}中读取Chroma向量数据库")
        
        self.bm25_retriever.load()
        print(f"成功从路径{self.bm25_retriever.persist_path}中读取bm25向量数据库")










    
    def repository_summary(self)->str:
        #1.只读实际Dense索引，旧索引也能用；不重新嵌入或改写历史文件
        if self._db is None:
            raise RuntimeError("chroma向量数据库未初始化，无法读取仓库简介")
        if self._repo_summary is not None:
            return self._repo_summary
        files = {}
        offset = 0
        while True:
            batch = self._db.get(include=["metadatas","documents"],limit=256,offset=offset)
            for meta,content in zip(batch["metadatas"],batch["documents"]):
                source = str((meta or {}).get("source","")).replace("\\","/")
                if not source:
                    continue
                names = files.setdefault(source,set())
                #索引块可能从函数中间开始，提取可见声明，不把碎片当完整Python解析
                names.update(re.findall(r"^[ \t]*(?:async[ \t]+)?(?:class|def)[ \t]+(\w+)",content or "",re.MULTILINE))
                files[source] = set(sorted(names)[:12])
            if len(batch["ids"]) < 256:
                break
            offset += 256
        #2.限制简介长度，没列出的文件仍可检索；同一索引只整理一次
        lines = ["项目：当前检索索引（文件与可见符号摘要，可能不全）"]
        size = len(lines[0])
        for source,names in sorted(files.items()):
            line = f"{source}：{'、'.join(sorted(names))}"
            if size + len(line) + 1 > 6000:
                lines.append("其余文件略；摘要未列出不代表仓库中不存在。")
                break
            lines.append(line)
            size += len(line) + 1
        if not files:
            lines.append("当前索引没有可用的文件来源信息。")
        self._repo_summary = "\n".join(lines)
        return self._repo_summary

    #内容相同也可能来自不同文件，来源和位置一起做身份标记
    @staticmethod
    def _chunk_key(doc : Document)->tuple:
        meta = doc.metadata
        return (str(meta.get("source","")).replace("\\","/"),meta.get("chunk_id"),
                meta.get("start_line"),meta.get("end_line"),doc.page_content)

    #将用户提问转为向量，找到库中最接近的k个代码块。返回List[Document]
    def search(self,query:str,top_k = 4)->List[Document]:
        #看向量数据库有没有
        if self._db is None:
            raise RuntimeError("chroma向量数据库未初始化，查询失败")
        if self.bm25_retriever is None:
            raise RuntimeError("bm25向量数据库未初始化，查询失败")


        #1.每路至少召回10个，不能少于要返回的 top_k
        recall_k = max(10,top_k)
        chroma_results = self._db.similarity_search(query,k=recall_k)#此处为k而非top_k
        bm25_results = self.bm25_retriever.search(query,top_k=recall_k)

        return self._rrf_fusion(chroma_results,bm25_results,top_k)

    def _rrf_fusion(self,chroma_results,bm25_results,top_k):
        #2.构建计分板，用代码块的来源、位置和内容做键
        doc_map : Dict[tuple,Document] = {}
        rrf_scores : Dict[tuple,float] = {}
        #平滑常数，减小相邻名次的分差
        c = 60
        #3.分别累加两路的 RRF 分数
        for results in (chroma_results,bm25_results):
            seen = set()
            for doc in results:
                content = self._chunk_key(doc)
                #同一路先去重，重复块不能挤占后面的名次
                if content in seen:
                    continue
                rank = len(seen)
                seen.add(content)
                doc_map[content] = doc
                #两路都出现就累加，没出现过就从 0.0 开始
                rrf_scores[content] = rrf_scores.get(content,0.0) + 1.0/(rank+c+1)

        #4.按总分降序，保留前 top_k 个原文档
        sorted_contents = sorted(rrf_scores,key=lambda x:rrf_scores[x],reverse=True)
        return [doc_map[content] for content in sorted_contents[:top_k]]

    #qwen接口的精排
    def _rerank_documents_qwen(self,query:str,docs:List[Document],top_k:int)->List[Document]:
        if not docs:
            return[]
        print(f"正在调用{self.reranker_name}模型对{len(docs)}个文档进行精排\n")
        #1.提取纯文字列表

        docs_text = [doc.page_content for doc in docs]

        #2.获取结果

        resp = dashscope.TextReRank.call(
            model=self.reranker_name,
            query=query,
            top_n=top_k,
            documents=docs_text,
            return_documents=False#返回前top_k排名索引，方便处理
        )

        #3.请求失败就中止，避免把粗排成绩标成重排成绩；网络异常直接向上传递
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"云端重排失败（{resp.status_code}）：{resp.message}")
        return [docs[item.index] for item in resp.output.results]




    #新增一个独立的rerank打分方法
    def _rerank_documents(self,query:str,docs:List[Document],top_k:int)->List[Document]:
        #if docs is None:
            #return[]

        #覆盖空列表情况
        if not docs:
            return []
        
        #0.组装query,document文本对
        sentence_pairs = [[query,doc.page_content] for doc in docs]
        
        print(f"正在调用{self.reranker_name}模型对{len(docs)}个文档进行精排\n")
        #1.交叉编码器打分   
        scores = self.reranker.predict(sentence_pairs)


        #2.将分数和文件绑定

        scored_docs = list(zip(docs,scores))  #大写List:类型注解；小写list:类型转换


        #3.排序

        scored_docs.sort(key = lambda x:x[1],reverse=True)

        #4.剥离分数，输出前top_k个document对象
        
        return [doc for doc,score in scored_docs[:top_k]]



    #为构建评估实验：构建专门的分支

    #直接构建接口
    def search_dense(self,query:str,top_k:int = 5)->List[Document]:
        docs, _, _ = self.search_dense_with_top1(query, top_k=top_k)
        return docs

    def search_dense_with_top1(
        self,
        query: str,
        top_k: int = 5,
        *,
        max_distance: Optional[float] = None,
    ) -> Tuple[List[Document], Optional[Document], Optional[float]]:
        """一次召回返回结果及粗排 top-1；超过距离阈值时跳过精排。"""
        if self._db is None:
            raise RuntimeError("chroma向量数据库未初始化，查询失败")

        recall_k = top_k * 3 if self.use_reranker else top_k
        pairs = self._db.similarity_search_with_score(query, k=recall_k)
        if not pairs:
            return [], None, None

        top1_doc, top1_distance = pairs[0]
        top1_distance = float(top1_distance)
        if max_distance is not None and top1_distance > max_distance:
            return [], top1_doc, top1_distance

        candidate_docs = [doc for doc, _ in pairs]
        if self.reranker is not None:
            docs = self._rerank_documents(query, candidate_docs, top_k)
        elif self.use_reranker:
            docs = self._rerank_documents_qwen(query, candidate_docs, top_k)
        else:
            docs = candidate_docs[:top_k]
        return docs, top1_doc, top1_distance

    def dense_top1_with_distance(self, query: str) -> Tuple[Optional[Document], Optional[float]]:
        """
        仅 Chroma Dense 粗排，取 top-1 文档及距离（与默认主链路一致，不经 Reranker）。
        距离含义以当前 Chroma / 嵌入空间为准：数值越大通常表示与库内片段越不相似，
        适合作为软熔断「召回质量差」的观测信号；阈值需在本机分布上标定。
        """
        if self._db is None:
            raise RuntimeError("chroma向量数据库未初始化，查询失败")
        pairs = self._db.similarity_search_with_score(query, k=1)
        if not pairs:
            return None, None
        doc, dist = pairs[0]
        return doc, float(dist)

    def search_bm25(self,query:str,top_k:int = 5)->List[Document]:
        recall_k = top_k * 3 if self.use_reranker else top_k
        candidate_docs = self.bm25_retriever.search(query, top_k=recall_k)

        if self.reranker is not None:   #本地 CrossEncoder
            return self._rerank_documents(query, candidate_docs, top_k)
        elif self.use_reranker:   #云端 qwen rerank
            return self._rerank_documents_qwen(query, candidate_docs, top_k)
        else:
            return candidate_docs[:top_k]
    def search_hybrid(self,query:str,top_k:int = 5)->List[Document]:

        if self.use_reranker:
            recall_k = top_k * 3
        else:
            recall_k = top_k
        
        #1.粗排
        candidate_docs = self.search(query,top_k=recall_k)

        #2.精排
        if self.reranker is not None:   #本地 CrossEncoder
            return self._rerank_documents(query,candidate_docs,top_k)
        elif self.use_reranker:   #云端 qwen rerank
            return self._rerank_documents_qwen(query,candidate_docs,top_k)





        else:
            return candidate_docs[:top_k]



        
