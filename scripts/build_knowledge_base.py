#建库
#用法（项目根目录）：
#  python scripts/build_knowledge_base.py                                   默认：给 src 建库
#  python scripts/build_knowledge_base.py --embedding qwen3.7-text-embedding-flash --persist-dir data/indices/src_flash
#  python scripts/build_knowledge_base.py --repo-path /path/to/local/repo --persist-dir data/indices/external_repo
#  python scripts/build_knowledge_base.py --chunk-mode ast --persist-dir data/indices/src_ast     按语法树切块，做对比
import argparse
import sys
from pathlib import Path

#直接 python scripts/xxx.py 运行时，把项目根目录加进路径
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.rag.splitter import CodeSplitter
from src.rag.loader import RePoLoader
from src.rag.vectorstore import VectorStore

def main() -> None:
    parser = argparse.ArgumentParser(description="RepoRouter 建库")
    parser.add_argument("--repo-path",default="src",help="要建库的仓库目录，可以是任意本地仓库")
    parser.add_argument("--persist-dir",default=settings.Chroma_persist_dir,help="向量库和bm25索引的新目录，不覆盖已有路径")
    parser.add_argument("--embedding",default=settings.qwen_embedding_name,help="qwen embedding 模型名")
    parser.add_argument("--chunk-mode",default=settings.code_chunk_mode,choices=["char","ast"],help="代码按字符切还是按语法树切")
    args = parser.parse_args()

    repo_path = Path(args.repo_path).expanduser().resolve()
    persist_dir = Path(args.persist_dir).expanduser()
    if not repo_path.is_dir():
        parser.error(f"源码目录不存在或不是目录：{repo_path}")
    if persist_dir.exists() or persist_dir.is_symlink():
        parser.error(f"索引目标已存在，为保护原有数据拒绝覆盖；请指定新的 --persist-dir：{persist_dir}")
    persist_dir = persist_dir.resolve()

    try:
        loader = RePoLoader(repo_path)
        print(f"正在从{repo_path}建立本地知识库（embedding：{args.embedding}）\n")

        docs = loader.load()
        splitter = CodeSplitter(
            code_chunk_size=settings.code_chunk_size,
            code_chunk_overlap=settings.code_chunk_overlap,
            text_chunk_size=settings.text_chunk_size,
            text_chunk_overlap=settings.text_chunk_overlap,
            mode=args.chunk_mode,
        )

        all_chunks = splitter.split(docs)
        if not all_chunks:
            raise ValueError("源码目录中没有可索引的非空文本，未创建索引")
        #打印块数和平均长度：两种切法对比时，块的大小要差不多才公平
        avg = sum(len(c.page_content) for c in all_chunks) / len(all_chunks)
        print(f"切块方式：{args.chunk_mode} | 共 {len(all_chunks)} 块 | 平均 {avg:.0f} 字符")

        # 独占创建目标，防止预检后其他进程创建的索引被追加写入。
        persist_dir.mkdir(parents=True, exist_ok=False)
        settings.Chroma_persist_dir = str(persist_dir)
        settings.bm25_persist_dir = str(persist_dir / "bm25_index.pkl")
        settings.qwen_embedding_name = args.embedding
        db = VectorStore()
        #切分好的词块给向量数据库导入
        db.index_document(all_chunks)
        print(f"本地知识库建立完成！保存在 {persist_dir}\n")
    except Exception as e:
        print(f"本地知识库建立失败...\n错误信息：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
