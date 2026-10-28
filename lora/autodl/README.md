# AutoDL：4B v2 路由微调与 Q4 部署

第二版已完成 Qwen3.5-4B LoRA v1（497 条）和 v2（637 条）；本页操作步骤对应 v2。数据与成绩见 [02 数据构建](../../评估/第二版/02-数据构建.md)、[03 路由微调](../../评估/第二版/03-路由微调.md)、[05 系统接入](../../评估/第二版/05-量化与系统接入.md)。

## 历史训练配置

以下合并 v1/v2 自动模型卡的共同记录，并用 [训练 YAML](train_qwen35_4b.yaml) 核对参数。

| 项目 | v1 / v2 留存记录 |
|---|---|
| adapter 与训练数据 | [v1](https://huggingface.co/Mrch1io/RepoAgent-Qwen3.5-4B-LoRA-v1)：497 条、无简介；[v2](https://huggingface.co/Mrch1io/RepoAgent-Qwen3.5-4B-LoRA-v2)：637 条、有简介 |
| 基座 / 数据集键 | `Qwen3.5-4B` / `repoagent_phase2a_routing` |
| LoRA | rank 16、alpha 32、dropout 0.05、target `all`，冻结视觉编码器 |
| 模板 / 截断 / 精度 | `qwen3_5_nothink` / 2048 token / BF16 |
| batch | 训练每卡 2、梯度累积 4、单卡有效 batch 8；模型卡记录 eval batch 8 |
| 学习率 / 轮次 / seed | 1e-4 / 3 epochs / 42 |
| 优化器 / 调度 | AdamW Torch Fused，betas=(0.9, 0.999)、epsilon=1e-8；cosine，**warmup_ratio=0.1** |
| 框架版本 | PEFT 0.18.1、Transformers 5.6.0、PyTorch 2.12.1+cu130、Datasets 4.0.0、Tokenizers 0.22.2 |
| 模型卡元数据 | library=`peft`、pipeline=`text-generation`、license=`other`；结果表为空，成绩以评测日志为准 |

自动卡把预热字段写成 `lr_scheduler_warmup_steps: 0.1`；训练 YAML 实际为 **10% 预热比例**。

## 1. 离线重建并核验 v2 数据

在本地项目根目录、已激活环境后执行；适用于 PowerShell / Bash，不调用模型 API：

```text
python -m experiments.routing_data.reproduce_data
```

脚本按 497 条复核数据＋100 条隐式查库题＋40 条概念题合并，逐条核对字段与顺序，忽略空白格式；全部一致才写入 `experiments/routing_data/outputs/reproduced/`。目录已存在时拒绝覆盖，可用 `--out-dir` 指定新目录。

| 输出 | 用途 |
|---|---|
| `ft_qa_train_v2.jsonl` / `routing_train.json` | 637 条原始样本 / 带简介的 ShareGPT 训练集 |
| `routing_train_v1_no_summary.json` | 497 条 v1 对照 |
| `eval_in_domain*.json` / `eval_cross_repo*.json` | 50 / 87 条测试，各保留有简介和无简介版 |
| `VERIFICATION.json` | 核验结果与输入 SHA256 |

需要留存训练数据，只有代码的 clone 不够。87 条跨仓库题及简介直接复制历史快照，不能用今天的源码重新生成简介后仍称为同条件复测。

## 2. 准备 AutoDL 环境

**第 2–5 节为 Linux Bash。** 将项目及上一步输出上传到 `/root/RepoRouter`；已有安装目录可复用。环境须支持 `Qwen3_5ForConditionalGeneration` 与 `qwen3_5_nothink`。

```bash
cd /root
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd /root/LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install modelscope
modelscope download --model Qwen/Qwen3.5-4B --local_dir /root/autodl-tmp/models/Qwen/Qwen3.5-4B
```

## 3. 训练 4B v2

以下复制数据并生成复现配置，关闭覆盖。训练配置与[导出配置](export_qwen35_4b_v2.yaml)均使用 `/root/autodl-tmp/repoagent/saves/qwen3.5-4b/lora/repoagent_phase2a_v2`；若已存在则停止，另开实验需同步更换训练、导出和评测中的 adapter 路径。

```bash
set -euo pipefail
cd /root/RepoRouter
python - <<'PY'
from pathlib import Path
import shutil
import yaml

root = Path('/root/RepoRouter')
run = Path('/root/autodl-tmp/repoagent/reproduced')
config = yaml.safe_load((root / 'lora/autodl/train_qwen35_4b.yaml').read_text())
if Path(config['output_dir']).exists():
    raise SystemExit('Adapter output already exists; choose a new experiment path first.')
data = run / 'data'
data.mkdir(parents=True, exist_ok=False)
for name in ('routing_train.json', 'eval_in_domain.json', 'eval_cross_repo.json'):
    shutil.copy2(root / 'experiments/routing_data/outputs/reproduced' / name, data / name)
shutil.copy2(root / 'lora/dataset_info.json', data / 'dataset_info.json')
config['dataset_dir'] = str(data)
config['overwrite_output_dir'] = False
(run / 'train_qwen35_4b_v2.yaml').write_text(yaml.safe_dump(config, allow_unicode=True))
PY
cd /root/LLaMA-Factory
llamafactory-cli train /root/autodl-tmp/repoagent/reproduced/train_qwen35_4b_v2.yaml
```

保留 `dataset_info.json` 的 `from/value` 与 `system/user/assistant` 映射。留存 v2 训练为 240 step、约 47.6 分钟；训练后检查 `train_results.json`、`trainer_log.jsonl` 和 loss 曲线。

## 4. 对照评测原版与 v2

两模型使用相同测试文件，默认带仓库简介。每次新运行使用独立日志目录。

```bash
set -euo pipefail
cd /root/RepoRouter
BASE=/root/autodl-tmp/models/Qwen/Qwen3.5-4B
ADAPTER=/root/autodl-tmp/repoagent/saves/qwen3.5-4b/lora/repoagent_phase2a_v2
DATA=/root/autodl-tmp/repoagent/reproduced/data
LOGS=/root/autodl-tmp/repoagent/reproduced/logs
mkdir -p "$LOGS"
for T in eval_in_domain eval_cross_repo; do
  python lora/eval_routing_base.py --base-model "$BASE" --test-data "$DATA/$T.json" | tee "$LOGS/${T}_base_4b.txt"
  python lora/eval_routing_lora.py --base-model "$BASE" --lora-path "$ADAPTER" --test-data "$DATA/$T.json" | tee "$LOGS/${T}_lora_4b_v2.txt"
done
```

复测 v1 时使用 `repoagent_phase2a` adapter 和 `*_no_summary.json`；API 基线用 `python lora/eval_routing_api.py --model deepseek-flash --test-data "$DATA/eval_in_domain.json"` 或 `python lora/eval_routing_api.py --model deepseek-v4-pro --test-data "$DATA/eval_in_domain.json"`，跨仓库评测将文件名换成 `eval_cross_repo.json`。解析失败计错，不从金标分母删除；同分布工具/非工具题为 31/19，跨仓库为 35/52。参数合法不等于检索或回答正确，完整口径见 [03 路由微调](../../评估/第二版/03-路由微调.md)。

## 5. 合并、导出与 Q4 量化

这是 BF16 LoRA **训练后的推理量化**。先确认导出目录与两份 GGUF 输出均不存在，已有模型另取新路径。

```bash
set -euo pipefail
cd /root/LLaMA-Factory
llamafactory-cli export /root/RepoRouter/lora/autodl/export_qwen35_4b_v2.yaml
cd /root/autodl-tmp
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
pip install -r llama.cpp/requirements.txt
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON
cmake --build llama.cpp/build --config Release -j
python llama.cpp/convert_hf_to_gguf.py /root/autodl-tmp/export/repoagent-router-4b-v2 --outfile /root/autodl-tmp/export/router-4b-v2-f16.gguf --outtype f16
./llama.cpp/build/bin/llama-quantize /root/autodl-tmp/export/router-4b-v2-f16.gguf /root/autodl-tmp/export/router-4b-v2-Q4_K_M.gguf Q4_K_M
```

## 6. 本地服务与端到端评测

下载 Q4 文件到本机。在 **Windows PowerShell、llama.cpp 可执行文件所在目录**启动支持该架构与 GPU 后端的服务：

```powershell
.\llama-server.exe -m C:/models/RepoRouter/router-4b-v2-Q4_K_M.gguf --port 8080 --jinja -ngl 99 --alias repoagent-router
```

项目环境配置如下；回答模型 API 仍需正常配置：

```dotenv
routing_backend=phase2a
router_base_url=http://127.0.0.1:8080/v1
router_model=repoagent-router
```

另开 PowerShell，在项目根目录激活环境、确认索引可用，执行 90 题 × 3 次。每次更换输出名，保留已有结果；此步骤会调用回答 API。

```powershell
python scripts/eval_agent.py --routing-backend phase2a --model deepseek-flash --repeat 3 --jsonl-out data/v2/reproduced/e2e_phase2a_v2_Q4.jsonl
python scripts/eval_agent.py --compare data/v2/e2e_deepseek-flash.jsonl data/v2/reproduced/e2e_phase2a_v2_Q4.jsonl
```

历史 270 次查库决策正确 260 次，该批回答评分仍为 `pending`。后续 frozen90 / clean90 两批各 90 题已完成独立评分，见 [05 系统接入](../../评估/第二版/05-量化与系统接入.md)和 [06 回答质量](../../评估/第二版/06-回答质量与复盘.md)。

## 7. 9B 扩展计划

9B 仅有原版基线，尚未完成 LoRA 或量化实验。9B 的[训练配置](train_qwen35_9b.yaml)和[导出配置](export_qwen35_9b.yaml)还没跑过。
