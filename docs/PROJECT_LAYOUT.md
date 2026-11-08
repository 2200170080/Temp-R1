# Temp-R1 目录说明

## 常用入口

- [部署记录](Temp-R1_部署手册.md)：保留历史内容；其中环境和训练结论需逐项实测核验。
- [GitHub 同步说明](GitHub同步说明.md)：服务器向 GitHub 的同步方式。
- `environment/`：已有环境快照，供排查依赖版本使用。

## 目录职责

| 路径 | 内容 |
|---|---|
| `scripts/` | 数据处理、训练、评测与同步入口 |
| `LLaMA-Factory/` | SFT 框架源码与其配置；保留 editable 安装位置 |
| `verl/` | GRPO 框架源码 |
| `search_r1/` | 检索与交互式生成源码 |
| `data/` | MultiTQ、TimelineKGQA 及加工后的训练/评测输入 |
| `corpus/` | 知识图谱语料和检索索引 |
| `outputs/` | 已有 SFT 轨迹，已纳入 Git 跟踪 |
| `logs/` | 已结束的建索引日志；后续新日志应写入此目录 |
| `LLaMA-Factory/saves/` | SFT 模型产物的现有约定位置 |
| `verl_checkpoints/` | GRPO 模型产物的现有约定位置 |
| `docs/` | 部署、同步、目录说明与环境记录 |
| `assets/` | 原论文的说明图片 |

`retrieval.log` 当前保留在项目根目录：检索进程正在写入，整理不干预服务。
`scripts/github-sync/autosync.log` 保留原路径，保持同步器的日志约定。

索引、parquet、原始训练集、SFT 轨迹、模型权重和安装元数据都不是待删垃圾。
`__pycache__` 与 `.pyc` 属可再生文件，本次保留并通过 Git 忽略规则隐藏。
本次仅整理目录与说明，不代表部署问题或 1.5B 训练配置已经修复。
