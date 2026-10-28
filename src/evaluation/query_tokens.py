"""题面符号抽取（与 scripts/audit_retrieval.py 一致）。"""
from __future__ import annotations

import re
from pathlib import Path

from src.rag.loader import RePoLoader


def load_source_text(src_root : Path)->str:
    #1.与建库共用加载器，题库和实验结果不作为源码证据，支持的后缀也保持一致
    return "\n".join(doc.page_content for doc in RePoLoader(src_root).load())

#题面里太泛的英文词，不参与「源码是否出现」
_SKIP_WORDS = frozenset(
    {
        "what", "where", "when", "which", "how", "does", "the", "and", "for", "with",
        "from", "that", "this", "your", "have", "been", "will", "into", "about",
        "fastapi", "python", "docker", "github", "json", "api", "http", "https",
        "streamlit", "chrom", "bm25", "query", "dockerfile", "actions", "lora",
    }
)


def interesting_tokens(question: str) -> list[str]:
    raw = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", question)
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        tl = t.lower()
        if len(t) < 3 or tl in _SKIP_WORDS:
            continue
        if len(t) < 5 and "_" not in t:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def extra_anchor_tokens(question: str) -> list[str]:
    """题面涉及 rewriter 等但单拆词匹配不到类名时补充。"""
    q = question
    extra: list[str] = []
    if re.search(r"rewriter|rewrite|改写", q, re.I):
        for t in ("query_rewriter", "Query_rewriter"):
            if t not in extra:
                extra.append(t)
    return extra
