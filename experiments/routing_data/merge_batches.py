"""将 ft_qa_100.jsonl 并入 ft_qa_450_train.jsonl，重写与 450 冲突的 sample_id，输出 ft_qa_500_train.jsonl。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments" / "routing_data" / "outputs"


def load_jsonl(p: Path) -> list[dict]:
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def max_suffix_by_bucket(rows: list[dict]) -> dict[str, int]:
    pat = re.compile(r"^(domain|far_general|misleading_edge)_(\d+)$")
    mmax = {"domain": 0, "far_general": 0, "misleading_edge": 0}
    for r in rows:
        sid = r.get("sample_id", "")
        m = pat.match(sid)
        if not m:
            continue
        b, n = m.group(1), int(m.group(2))
        mmax[b] = max(mmax[b], n)
    return mmax


def next_sample_id(bucket: str, counters: dict[str, int]) -> str:
    counters[bucket] += 1
    return f"{bucket}_{counters[bucket]:04d}"


def main() -> None:
    #1.先保护已有合并结果，再读取输入
    p450 = OUT_DIR / "ft_qa_450_train.jsonl"
    p100 = OUT_DIR / "ft_qa_100.jsonl"
    out = OUT_DIR / "ft_qa_500_train.jsonl"
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"拒绝覆盖已有合并数据：{out}")

    r450 = load_jsonl(p450)
    r100 = load_jsonl(p100)
    mmax = max_suffix_by_bucket(r450)
    counters = {
        "domain": mmax["domain"],
        "far_general": mmax["far_general"],
        "misleading_edge": mmax["misleading_edge"],
    }

    renamed: list[dict] = []
    for r in r100:
        r2 = dict(r)
        b = r2["bucket"]
        old_sid = r2.get("sample_id")
        r2["sample_id"] = next_sample_id(b, counters)
        meta = dict(r2.get("_meta") or {})
        meta["_merged_from"] = "ft_qa_100.jsonl"
        meta["previous_sample_id"] = old_sid
        r2["_meta"] = meta
        renamed.append(r2)

    merged = r450 + renamed
    #2.独占创建，防止检查后出现的文件被覆盖
    with out.open("x",encoding="utf-8") as f:
        f.write("\n".join(json.dumps(x, ensure_ascii=False) for x in merged) + "\n")
    print(f"450_train={len(r450)} + pilot_100={len(renamed)} -> {len(merged)} -> {out}")
    print("max suffix in 450_train by bucket:", mmax)


if __name__ == "__main__":
    main()
