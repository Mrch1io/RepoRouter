from __future__ import annotations #引入未来特性
from langchain_core.documents import Document        #使用lanchain_core框架 RAG数据流，使用标准生态，以免冲突。
import os #os.walk遍历文件所需
import logging
from typing import List,Iterable      #类型标注
from pathlib import Path    #路径处理


class RePoLoader:
    """代码仓库加载器：
    输入仓库路径
    自动读取所有代码
    返回List[Document]，每个Document:全文内容+元数据
    
    
    """
    #加载前过滤：支持的文件类型
    SUPPORTED_EXTENSIONS = {
        ".py",".js",".txt",".java",".go",".cpp",".md",".rs",".yml",".yaml",".json"   #.java 之前少了点，永远匹配不上
    }



    #加载前过滤：忽略的目录

    IGNORE_DIRS = {
        #.开头：隐藏文件
        ".git",".idea",".venv","venv","__pycache__",".vscode","node_modules", "dist", "build",
        ".pytest_cache","archive","saves","评估"
    }

    #加载前过滤：忽略的文件
    #评测考题和标准答案，进了索引检索会直接搜到考题本身；评测脚本（eval_retrieval.py等）是正常代码，照样建库
    IGNORE_FILES = {
        "soft_fuse_dataset.py","e2e_manifest.py","rag_dataset.py","rewriter_noisy_dataset.py",
        "Q4评测复盘.md"
    }

    EVIDENCE_DIRS = {("data","v2"),("data","reports"),("data","audit"),
                     ("data","soft_fuse"),("experiments","routing_data","outputs")}

    def _is_experiment_data(self,path : Path)->bool:
        #1.只检查建库根目录里面的路径，显式选择的源码目录不能被父目录误伤
        parts = path.relative_to(self.repo_path).parts
        if any(parts[i:i+len(prefix)] == prefix for prefix in self.EVIDENCE_DIRS
               for i in range(len(parts)-len(prefix)+1)):
            return True
        if path.suffix in {".json",".jsonl"}:
            return path.name.startswith(("routing_train","eval_in_domain","eval_cross_repo",
                                         "e2e_","judge_","ft_qa_","soft_fuse_","chroma_verified_"))
        return False

    #1.初始化：路径:仓库根目录                 |表示两种类型均可
    def __init__(self,repo_path : str | Path):
        self.repo_path = Path(repo_path).resolve()  #封装为Path类方便路径处理。 
                                                            #.resolve定位到物理绝对路径，防止传入"."这样的情形
    
    #2.文件遍历：返回需要的文件 ！的路径！可迭代对象列表 标注返回类型
    def iter_file(self) -> Iterable[Path]:
        #遍历返回：绝对路径，文件夹列表，文件列表
        for root,dirs,files in os.walk(self.repo_path):
            #原地修改，否则如果做一个新的dirs，根据py指针特性，旧的在遍历的不会被修改。
            #去掉不需要的文件夹
            dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS and not self._is_experiment_data(Path(root) / d)]
            #去掉不需要的文件
            for fname in files:
                #先封装成Path类型
                path = Path(root) / fname
                if path.suffix in self.SUPPORTED_EXTENSIONS and fname not in self.IGNORE_FILES and not self._is_experiment_data(path):
                    yield path
    #3.读取路径列表，读取内容，封装打包

    def load(self, *, show_progress: bool = False) -> List[Document]:
        paths = list(self.iter_file())
        file_iter = paths
        if show_progress:
            try:
                from src.rag.tqdm_util import tqdm

                file_iter = tqdm(paths, desc="加载文件", unit="file")
            except ImportError:
                pass
        docs: List[Document] = []
        #1.遍历读取:
        for file_path in file_iter:
            try:
                text = file_path.read_text(encoding="utf-8",errors="ignore")#忽略乱码
            #读不到就记录原因，再进入下个文件；程序自身的异常仍向上抛出
            except OSError as e:
                logging.warning("读取文件失败，已跳过：%s（%s）",file_path,e)
                continue

            #获取相对路径：下一台电脑也能用：现在的文件绝对路径-开始的self.repo_path根目录
            rel_path = file_path.relative_to(self.repo_path)
            #2.构建元数据
            metadata = {
                #来源，文件名，后缀
                #rel_path是一个复杂的path类型，但是数据库在读取的时候接受不了，转为str
                "source" : str(rel_path),
                "file_name" : file_path.name, #Path类封装的好处：各种需要调用的数据变为成员变量
                "suffix" : file_path.suffix,

            }

            #将内容和元数据打包成标准生态Document类
            doc = Document(page_content = text,metadata = metadata)
            docs.append(doc)
        return docs

            
