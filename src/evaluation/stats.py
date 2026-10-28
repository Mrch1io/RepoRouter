#评测用的几个统计小工具：样本少的时候，光报一个百分比不够，要带上95%置信区间
import math
import random


#计算top_k命中率以及RR

def hit_and_rr(docs : list,expected_source):
    for idx,doc in enumerate(docs):
        #如果预期文件源中包含给定文件的源，就算数
        #Langchain中Document的底层源码，值可以是任意类型。显式再强制转成str形式，告诉编译器是字符串
        src = str(doc.metadata.get("source","")).replace("\\","/")   #对路径在win上替换
        if any(exp in src for exp in expected_source):
            return 1,1.0 / (idx + 1)    #加括号！

    #全部没有才返回0.。
    return 0,0.0


#1.比例的Wilson置信区间（比如“90题里对了85题”）：样本少时比正态近似靠谱
def wilson_ci(k : int,n : int,z : float = 1.96)->tuple:
    if n == 0:
        return 0.0,0.0
    p = k / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return max(0.0,center - half),min(1.0,center + half)


#2.均值的bootstrap置信区间（比如MRR）：有放回地重抽样很多次，看均值的波动范围
def bootstrap_ci(values : list,n_boot : int = 2000,seed : int = 42)->tuple:
    if not values:
        return 0.0,0.0
    rng = random.Random(seed)  #固定种子，结果可复现
    means = sorted(sum(rng.choices(values,k=len(values))) / len(values) for _ in range(n_boot))
    return means[int(0.025*n_boot)],means[int(0.975*n_boot) - 1]


#3.McNemar精确检验：同一批题，两个模型对错不一样的那部分，差异是不是偶然的
#b = A对B错的题数，c = A错B对的题数；返回双侧p值（越小越说明不是偶然）
def mcnemar_p(b : int,c : int)->float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b,c)
    p = sum(math.comb(n,i) for i in range(k + 1)) / 2**n
    return min(1.0,2*p)


#4.ROC-AUC：随便拿一个“该拦”的和一个“不该拦”的，前者距离更大的概率（1=完全分开，0.5=瞎猜）
def roc_auc(pos : list,neg : list)->float:
    if not pos or not neg:
        return float("nan")
    wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


#5.Cohen's kappa：两个评委的一致程度，扣掉“碰巧一致”的部分（1=完全一致，0=和瞎猜一样）
#a、b 是两个评委对同一批题的判断列表，比如 ["correct","partially_correct",...]
def cohen_kappa(a : list,b : list)->float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x,y in zip(a,b) if x == y) / n                         #实际一致率
    labels = set(a) | set(b)
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)         #碰巧一致的概率
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
