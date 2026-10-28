#两个评委对比：qwen3.8-max（默认评委）vs 第二评委（默认 deepseek-v4-pro），看同一批回答两家判得一不一致
#1.一致率 + Cohen's kappa（扣掉碰巧一致）  2.自评偏心：DeepSeek 评委给 DeepSeek 回答的分是不是偏高
#3.意见不一致的题导出成 md，给人工抽查
#
#用法（项目根目录，先用 judge_answers.py --judge-model deepseek-v4-pro 跑完第二评委）：
#  python scripts/compare_judges.py
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from src.config import settings
from src.evaluation.stats import cohen_kappa
from src.evaluation.llm_judge_e2e import load_jsonl_records

V2 = ROOT / "data" / "v2"


def main()->None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge2",default="deepseek-v4-pro")
    ap.add_argument("--answerers",nargs="+",default=["deepseek-flash","deepseek-v4-pro"])
    ap.add_argument("--out",type=Path,default=V2 / "judge_compare.md",help="输出新报告，不覆盖已有文件")
    args = ap.parse_args()
    if args.out.exists() or args.out.is_symlink():
        ap.error(f"输出文件已存在，请用 --out 指定新文件：{args.out}")
    j1,j2 = settings.judge_model,args.judge2

    lines = [f"# 评委对比：{j1} vs {j2}\n"]
    disagree = []
    for m in args.answerers:
        records = [load_jsonl_records(path) for path in (
            V2 / f"judge_e2e_{m}.jsonl", V2 / f"judge_e2e_{m}__by_{j2}.jsonl")]
        a,b = ({key:r for key,r in rows.items() if r.get("final_judgement") in {"correct","partially_correct","incorrect"} and not r.get("error")} for rows in records)
        ids = sorted(set(a) & set(b))
        if not ids:
            ap.error(f"{m} 没有可按 (sample_id, repeat) 配对的有效评分")
        #1.相同题号和轮次还需来自同一份回答；旧日志没有指纹时按键配对
        if any(a[i].get("input_sha256") and b[i].get("input_sha256") and a[i]["input_sha256"] != b[i]["input_sha256"] for i in ids):
            ap.error(f"{m} 两份评分的 input_sha256 不一致")
        fa,fb = [a[i]["final_judgement"] for i in ids],[b[i]["final_judgement"] for i in ids]
        #二分类口径：correct vs 其他
        ca,cb = [x == "correct" for x in fa],[x == "correct" for x in fb]
        sa,sb = [a[i]["score"] for i in ids],[b[i]["score"] for i in ids]
        diff = [y - x for x,y in zip(sa,sb)]
        lines += [
            f"\n## 被评模型：{m}（{len(ids)} 条配对记录，{len({i[0] for i in ids})} 道题）\n",
            f"- 评分失败或无有效判定：{j1} {len(records[0])-len(a)} 条，{j2} {len(records[1])-len(b)} 条。",
            f"- 未配对的有效评分：{j1} {len(a)-len(ids)} 条，{j2} {len(b)-len(ids)} 条；repeat 从 0 开始。",
            "",
            f"| 评委 | correct | partially_correct | incorrect | 平均分 |\n|---|---|---|---|---|",
            *[f"| {j} | {f.count('correct')} | {f.count('partially_correct')} | {f.count('incorrect')} | {sum(s)/len(s):.2f} |" for j,f,s in ((j1,fa,sa),(j2,fb,sb))],
            f"\n- 三分类一致：{sum(x == y for x,y in zip(fa,fb))}/{len(ids)}，kappa = {cohen_kappa(fa,fb):.3f}",
            f"- 只看“correct 与否”：一致 {sum(x == y for x,y in zip(ca,cb))}/{len(ids)}，kappa = {cohen_kappa(ca,cb):.3f}",
            f"- 分数差（{j2} − {j1}）：平均 {sum(diff)/len(diff):+.2f}，{j2} 更高 {sum(d > 0 for d in diff)} 条、更低 {sum(d < 0 for d in diff)} 条",
        ]
        #意见不一致：correct 与否判得不一样的题
        for i in ids:
            if (a[i]["final_judgement"] == "correct") != (b[i]["final_judgement"] == "correct"):
                disagree.append((m,i,a[i],b[i]))

    question_count = len({key[0] for _, key, _, _ in disagree})
    lines.append(
        f"\n## 意见不一致的记录（correct 与否判得不一样，共 {len(disagree)} 条记录，"
        f"涉及 {question_count} 道不同题，建议人工抽查）\n"
    )
    for m,i,x,y in disagree:
        lines += [f"### {m} · `{i[0]}` · repeat={i[1]} · {x.get('bucket_type')}\n", f"**问题**：{x.get('question')}\n",
                  f"- **{j1}**：{x['final_judgement']}（{x['score']} 分）— {str(x.get('reason',''))[:300]}",
                  f"- **{j2}**：{y['final_judgement']}（{y['score']} 分）— {str(y.get('reason',''))[:300]}\n"]

    out = args.out
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if l.startswith(("##","- ","|"))))
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
