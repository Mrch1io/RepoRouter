# RepoAgent 评估文档

本目录包含 RepoAgent 项目的各类评估实验记录与结论。

## 文档索引

| 文件 | 内容 |
|------|------|
| [01-检索引擎基线](./01-检索引擎基线.md) | Dense / BM25 / Hybrid 三种检索引擎基线对比 |
| [02-候选池大小对比](./02-候选池大小对比.md) | Hybrid 检索中 pool_size（10 vs 20）调参实验 |
| [03-重排消融](./03-重排消融.md) | BGE-base、GTE-v2、不精排 消融对比 |
| [04-查询改写消融](./04-查询改写消融.md) | Query Rewrite 对检索 MRR 的影响 |
| [05-LoRA评测](./05-LoRA评测.md) | LoRA 微调后 JSON 遵循率、工具路由准确率等 |
| [06-多工具协作](./06-多工具协作.md) | Agent 多工具串联（Tool Chaining）示例 |
| [07-向量库实现](./07-向量库实现.md) | VectorStore 类实现摘要（来自 search_codebase 输出）|
| [08-软熔断与端到端评测](./08-软熔断与端到端评测.md) | Dense 距离标定 + 全链路 90 条快照（`data/` 落盘）；E2E 第一层 gold 见 `e2e_manifest.py`（`grounding_tier`） |
| [09-提示词优化复测](./09-提示词优化复测.md) | 优化 prompt / `search_codebase` 描述后 E2E 对照；离线重算（`rescore_e2e_jsonl.py`）+ **bad case**（`analyze_e2e_badcase.py` → `data/reports/e2e_badcase_report_*.md`） |
| [10-回答质量评分](./10-回答质量评分.md) | E2E **第三层**：按桶标准 + 统一四项；抽样 22 条；`scripts/eval_llm_judge_e2e.py` → `data/eval/e2e_llm_judge_phase1.jsonl` |
| [11-路由微调](./11-路由微调.md) | Phase 2A **单轮路由** SFT：数据/ShareGPT/评测指标；Base vs LoRA vs API（50 条） |
| [12-进展与计划](./12-进展与计划.md) | **咨询口述**：**三关键点**（检索闭环 · 飞轮+路由SFT · RAGAS/Self-RAG 计划） |
| [12-进展展示.html](./12-进展展示.html) | **一页展示**（浏览器打开）：关键数据表、Phase1/E2E/Phase2A 数字 |

## 检索模块结论摘要

- **默认引擎**：纯 Dense（Qwen text-embedding-v4）
- **不启用**：Reranker、Hybrid（BM25 在代码语料中引入噪声）
- **Query Rewrite**：对口语化 query 有显著提升（MRR +16.5%）
