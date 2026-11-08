# Temp-R1 目录与产物说明

## 当前进度

正式 Qwen2.5-1.5B SFT 已完成，GRPO 两步测试通过。
正式 Stage1 在第一步反向传播时显存不足，现已暂停；Stage2 和正式评测尚未完成。
本次整理只处理目录和旧产物，不改变训练参数、模型权重或现有检索服务。

## 常用入口

| 用途 | 路径 |
|---|---|
| 1.5B SFT | `scripts/multitq/train_sft_qwen_1p5b.sh` |
| 1.5B GRPO Stage1 | `scripts/multitq/train_grpo_stage1_1p5b.sh` |
| 1.5B GRPO 两步测试 | `scripts/multitq/train_grpo_stage1_1p5b_smoke.sh` |
| 已安装的 GRPO 显存补丁工具 | `scripts/maintenance/fix_grpo_response_logits.py` |
| 本次整理工具（默认预览） | `scripts/maintenance/cleanup_tempr1.py` |
| 目录整理旧工具 | `scripts/maintenance/organize_tempr1.py` |
| 时序 KG 转换与只读核对 | `scripts/data_process/convert_kg_to_jsonl.py` |
| 检索服务 | `scripts/retrieval/retrieval_launch.sh` |
| GitHub 自动同步 | `scripts/github-sync/autosync.sh` |

上述训练入口仅作定位。本次不运行训练；原作者其他脚本继续保留供参考。

## 目录职责

```text
Temp-R1/
├── scripts/                 数据处理、按数据集训练/评测、维护及同步工具
│   ├── multitq/             MultiTQ 入口，包括当前 1.5B 配置
│   ├── timelinekgqa/        原作者 TimelineKGQA 入口
│   ├── data_process/        数据、轨迹筛选与转换
│   ├── maintenance/         整理与修复工具
│   └── github-sync/         同步脚本及其活动日志
├── docs/                    部署记录、目录说明、同步说明、环境快照
├── logs/                    正式 SFT 成功日志与最近正式 GRPO 失败日志
│   └── grpo_runs/           以后运行的 Hydra 配置；旧测试记录已清理
├── outputs/                 作者提供的 GPT 轨迹和 SFT 中间数据
├── data/                    原始数据、分类数据及 GRPO parquet
├── corpus/                  知识图谱语料及 kg_index 检索索引
├── LLaMA-Factory/           SFT 框架、配置及正式 SFT 权重
├── verl/                    GRPO 框架，保留已验证的源码修复
├── search_r1/               检索与交互生成源码
├── verl_checkpoints/        后续正式 GRPO 权重的约定位置
└── assets/                  论文说明图片
```

保留 editable 安装的框架目录位置。正式权重和数据不为美化目录而搬迁。
检索服务正在使用的 `logs/retrieval.log` 与同步器的
`scripts/github-sync/autosync.log` 继续保留原位。

## 保留与清理边界

- 保留最终 SFT：`LLaMA-Factory/saves/qwen2.5-1.5b-it/full/sft/`
  中的独立权重、config、tokenizer、训练结果及 loss 图。
- 旧模型清理范围为已列明的 `checkpoint-50/`、`checkpoint-100/`、
  `sft-smoke-20261108-114928/` 和
  `qwen1p5b-stage1-smoke-20261108-150209/`。
  删除前检查最终 SFT 的配置、tokenizer 和 safetensors 文件结构。
- 同时清理本次建索引、预检、显存补丁与 SFT/GRPO 测试日志，以及
  `logs/grpo_runs/2026-11-08/` 下的
  `14-46-41/`、`15-02-24/`、`15-13-15/` 历史配置目录。
  保留正式 `sft_qwen_1p5b.log` 与最近的 `grpo_stage1_1p5b.log`。
- 删除已结束的 SFT TensorBoard 事件文件及其空目录；
  最终模型目录中的训练指标、`trainer_log.jsonl` 和 loss 图继续保留。
- 删除旧 SFT 检查点也会删除其中的优化器状态，不能再从第 50/100 步精确续训；
  独立最终 SFT 权重继续用于后续 GRPO 和推理。
- 保留所有未明确列入删除清单的模型目录，尤其正式 GRPO 的 `actor/global_step_*`。
- 保留作者轨迹、原始/加工数据、索引、关键成功/失败日志、安装元数据和缓存。
- 当前 1.5B GRPO 入口的新配置写到 `logs/grpo_runs/<experiment_name>/`。
  三套已结束的历史运行配置会实际删除，不再转存到归档目录。
- 已核实的阶段结论：正式 SFT 完成 100 步、2 个 epoch，平均训练 loss 约
  0.14289；响应 logits 的显存修复已通过 CPU 输出/梯度等价检查及两步
  GRPO 测试；正式 Stage1 在首步 `loss.backward()` 显存不足，
  当时剩余 1.29 GiB、还需分配 2.84 GiB。两步测试不代表正式评测结果。

## 语料与检索一致性

转换器从项目位置定位默认输入，支持显式输入/输出及 tab、pipe、空白四列格式。
每条输出包含字符串 `id/contents/subject/predicate/object/date`；`contents` 保持
`subject predicate object on YYYY-MM-DD`，保留关系下划线、原始顺序和重复事实。
日期必须是合法的 `YYYY-MM-DD`。坏行会使转换整体失败；已有输出不覆盖，新文件完整成功后才发布。
旧入口 `scripts/convert_kg_to_jsonl.py` 已由规范位置的实现替代。

```bash
python scripts/data_process/convert_kg_to_jsonl.py --verify-against corpus/full_kg.jsonl
```

上述核对逐行比较六个字段及记录数量，不生成语料、不修改索引，也不加载 GPU。
当前 `full.txt` 与 `full_kg.jsonl` 各有 461329 条记录，已通过六字段逐行全量核对；48 项回归测试及五项实际接口验收均通过。
FAISS 的检索位置依赖 JSONL 行顺序；行数、顺序或 `contents` 变化后应在新目录重建索引。
行数相等本身不能证明旧索引与新语料兼容。

检索接口默认 `return_scores=false` 返回文档列表，`true` 返回带分数的文档包装。
空查询返回空结果；`topk` 必须为正整数，并限制到索引实际大小。
空索引或索引/语料行数不匹配时，服务在加载 GPU 模型之前停止并报出原因。

## 整理后的检查

整理工具默认仅输出计划；`--apply` 会执行已列明的本机操作。
应先暂停本项目的 GitHub `sync` 会话；工具遇到仍在运行的训练或同步进程会停止。
完成后核对 `git diff --stat`、最终 SFT 和检索服务，再恢复同步。
本次操作不会提交、推送或启动实验。
