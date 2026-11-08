# Temp-R1 复现手册

> 目标：在单卡 NVIDIA RTX A6000（48 GB）上复现 Qwen2.5-1.5B 的 Temp-R1 流程。
> 更新日期：2026-11-08，Asia/Shanghai。代码审核固定在 `main` 的 `fd28864b4229287e19fb3c6c7451ce0ed224c405`，提交时间为 2026-11-08 18:32:22。
> 运行根：`/root/user/cs_tcci_zengdexin/Temp-R1`，下文用 `$HOME/Temp-R1` 表示。服务器为 `ecs-instance-cdb4917c`，SSH 别名为 `A6000x1-6-168`。
> 当前结论：正式 SFT 已完成；GRPO 两步短跑通过；正式 Stage1 首步反向传播显存不足，按用户要求暂停。Stage2 与统一正式评测未取得完成证据。**本次更新手册不启动训练。**

## 0. 全流程与当前进度

作者图对应的顺序是：教师模型与时序知识图谱交互生成轨迹 → 正确性与格式筛选 → SFT Cold Start → 带检索交互的 GRPO → 评测。逆课程在 GRPO 中先训练复杂 multiple 问题，再转向完整训练集。SFT 与 GRPO 均使用本地 1.5B 模型。

本次直接复用作者提交的轨迹和 ShareGPT 数据，跳过重新调用 GPT API。复用教师数据不等于本次亲自生成，也不能仅凭文件名证明合并数据的教师模型版本。

| 阶段 | 前置条件 | 已有产物检查 / 跳过条件 | 输出与完成判据 | 当前证据与耗时 |
|---|---|---|---|---|
| 代码与环境 | Linux、Miniconda、CUDA 驱动、项目代码 | 导入及实际阶段通过则复用 | `tempr1` 负责 SFT/GRPO；保留已部署检索环境 | 已部署；全新安装耗时未验证 |
| 问题数据 | MultiTQ 官方题目 | 六份 parquet 行数正确则跳过重建 | Stage1 读 `train_multiple.parquet` | 已现场核对 103305 条训练、15720 条对应测试 |
| KG 与索引 | 四列时序事实、E5 编码器 | 语料逐行一致、索引配套则复用 | 461329 条语料；FAISS 768 维索引 | 全量只读核对通过；本次不重建索引 |
| 检索服务 | 配套索引、既有 `retriever` | 功能请求通过则跳过启动 | `POST 127.0.0.1:8000/retrieve` | 五项实际接口验收通过 |
| 教师轨迹与转换 | 作者数据，或修复后的生成器 | 作者轨迹及 SFT 数据可读则跳过 API | `outputs/*.jsonl` → LF ShareGPT 数据 | 合并轨迹 1055 条；本次没有重新付费生成 |
| SFT 冷启动 | Qwen2.5-1.5B、1000 条上限、Qwen 模板 | 权重、tokenizer 与完成日志通过则跳过 | 最终 `full/sft/` 权重 | 100 优化步、2 epoch；实测 49 分 58.65 秒 |
| GRPO 短跑 | 最终 SFT、检索服务 | 链路已验证且源码未变时无需重复 | 2 次更新与保存验证 | 已通过；旧短跑权重已清理 |
| GRPO Stage1 | 资源足够、用户决定恢复训练 | 短跑通过不能替代正式运行 | 计划 80 次外层更新，每 10 步保存 | 首步反向 OOM，暂停；无完成耗时 |
| GRPO Stage2 | Stage1 权重、单卡与评分适配 | 不能直接运行作者原入口 | 从 Stage1 权重初始化的后续训练 | 未取得完成证据；仍待适配 |
| 统一评测 | Base/SFT/GRPO 权重、同一协议 | loss 或 24 题验证不能替代 | 分模型、分题型正式指标与预测 | 未取得完成证据；入口待适配 |

“已通过”指本会话输出或固定版本日志覆盖的运行，不代表此刻资源、PID 和文件状态永久不变。下文默认在**服务器 Bash** 中执行；Windows 只用于登录和保存本手册。命令按小节有条件执行，检查失败先处理，不继续后续写入。预计耗时没有实测依据时不作承诺。

## 1. 连接、工作目录与资源

### 1.1 登录与现场检查

在已接入校园网的本机终端执行：

```powershell
ssh A6000x1-6-168
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `ssh A6000x1-6-168` | 使用已有 SSH 配置登录，不在手册保存密码 |

助手无法接入当前校园网服务器，服务器状态来自用户现场输出；GitHub 用于读取已同步源码和日志。在服务器执行：

```bash
hostname
printf '%s\n' "$HOME"
nvidia-smi
df -h "$HOME" /tmp /dev/shm
free -h
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `hostname` | 确认实际节点 |
| 2 | `printf ... "$HOME"` | 确认用户家目录，避免误把 `/root` 当项目用户目录 |
| 3 | `nvidia-smi` | 查看总显存、资源与可见进程 |
| 4 | `df -h ...` | 区分项目共享盘、临时盘和共享内存容量 |
| 5 | `free -h` | 检查 CPU offload 可用内存 |

历史 GPU 为 RTX A6000、49140 MiB，驱动 550.54.15，驱动报告 CUDA 12.4；训练 PyTorch 使用 cu121，两者版本不必相同。历史 CPU/内存和磁盘可用量不作为启动保证。

共享或容器服务器的 GPU 进程列表可能不完整。总显存占用不能证明占用者身份，只处理已确认属于自己的任务。当前继续等待资源，不能因 GPU 利用率低就自动恢复 GRPO。

### 本节验收

| 检查命令/检查项 | 判据 |
|---|---|
| 节点、`$HOME` | 符合本次用户与部署路径 |
| GPU、CPU 内存、项目盘和临时盘 | 资源足够；不终止归属不明的进程 |
| 当前训练状态 | 暂停要求仍有效，本节只检查 |

## 2. 代码获取、入口与目录职责

### 2.1 已有仓库：先核对，跳过克隆

```bash
cd "$HOME/Temp-R1"
git status --short
git rev-parse HEAD
git remote -v
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/Temp-R1"` | 固定运行根 |
| 2 | `git status --short` | 识别未提交改动，不擅自覆盖 |
| 3 | `git rev-parse HEAD` | 记录服务器当前版本，与本手册基线区分 |
| 4 | `git remote -v` | 区分工作仓库与作者上游 |

工作仓库为 [2200170080/Temp-R1](https://github.com/2200170080/Temp-R1)，作者上游为 [zjukg/Temp-R1](https://github.com/zjukg/Temp-R1)。现有服务器 `backup` 向用户仓库推送，`origin` 为作者上游参考；新克隆的 remote 名称可能不同，不能照搬。

### 2.2 新机器且目录不存在时获取适配版

```bash
cd "$HOME"
test ! -e "$HOME/Temp-R1" && git clone https://github.com/2200170080/Temp-R1.git "$HOME/Temp-R1"
cd "$HOME/Temp-R1"
git rev-parse HEAD
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME"` | 在用户目录准备工作区 |
| 2 | `test ! -e ... && git clone ...` | 仅在目标不存在时克隆，不覆盖现有部署 |
| 3 | `cd "$HOME/Temp-R1"` | 进入仓库 |
| 4 | `git rev-parse HEAD` | 记录实际版本；新 HEAD 可能晚于本文基线 |

普通部署不重建 Git 历史、不清空 Git 索引、不强推 main。不要用作者上游脚本覆盖当前适配版。

### 2.3 当前结构

```text
Temp-R1/
├── assets/                       论文原图，含 result*.png
├── corpus/                       KG 原文、JSONL、kg_index 索引
├── data/                         原题、题型划分、候选、GRPO parquet
├── docs/                         目录说明、历史部署、环境与同步记录
├── LLaMA-Factory/                SFT 框架、配置、数据与最终 SFT
├── logs/                         正式 SFT、GRPO、活动检索日志
├── outputs/                      作者轨迹等训练输入，不能按旧结果删除
├── scripts/
│   ├── data_process/             数据、轨迹与 KG 转换
│   ├── retrieval/                检索启动入口
│   ├── multitq/                  当前 1.5B 与作者参考入口
│   ├── timelinekgqa/             作者其他数据集参考入口
│   ├── maintenance/              修复与整理工具
│   └── github-sync/              同步脚本及活动日志
├── search_r1/                    检索和多轮交互实现
├── tests/                        转换与检索合同回归测试源码
├── verl/                         本项目修改版 GRPO 框架
├── requirements.txt              根依赖清单，含旧 transformers 上限
├── README.md
├── .gitignore
└── LICENSE
```

`verl_checkpoints/` 是未来 GRPO 保存位置；旧空目录已删除，保存时自动重建。Git 不跟踪空目录。`tests/` 是本次修复新增的回归测试源码，应保留。

editable 安装依赖框架原位置，正式权重、语料和索引不为美化目录搬迁。`requirements.txt` 留在根目录；检索入口已归入 `scripts/retrieval/`，活动日志为 `logs/retrieval.log`。

### 本节验收

| 检查命令/检查项 | 判据 |
|---|---|
| SHA 与 remote | 记录实际值，理解单向同步 |
| 当前训练入口 | 使用 `*_1p5b.sh`，不误用作者多卡脚本 |
| tests / outputs / assets | 按源码、输入、论文图片保留 |

## 3. 环境：复用现有部署，按需重建

### 3.1 环境约定与已验证版本

SFT 与 GRPO 共用 `tempr1`，不另建 SFT 环境。现有检索进程使用此前已部署的 `retriever`，本次保留。训练保持单环境不等于服务器只有一个 conda 环境；本次不迁移活动检索服务。

| 组件 | 基线 | 依据与边界 |
|---|---|---|
| Python | 训练 3.9；现有检索 3.10 | 两套现有解释器，不默认新建环境 |
| PyTorch / CUDA runtime | 2.4.0+cu121 / 12.1 | 已完成 SFT 和 GRPO 短跑 |
| vLLM | 0.6.3 | 安装元数据与本项目适配代码 |
| Transformers / tokenizers | 4.49.0 / 0.21.1 | 已替换导致 SFT 导入失败的 4.47.1 |
| FlashAttention / DeepSpeed | 2.6.3 / 0.15.4 | 既有 wheel 与 ZeRO-3 部署 |
| NumPy / datasets | 1.26.4 / 4.0.0 | 历史环境记录及实际阶段使用 |
| accelerate / tensordict | 1.10.1 / 0.5.0 | 历史环境记录 |
| LLaMA-Factory | 仓库内 0.9.4 开发版 | editable 安装当前子目录，不任意替换版本 |
| 检索 Faiss | 1.8.0 | 既有 GPU FAISS / E5 索引 |
| HTTP 组件 | 训练测试：FastAPI 0.128.8 / Pydantic 2.10.6 / httpx 0.28.1；检索：FastAPI 0.162.2 / Pydantic 2.14.3 | 两个环境的现场记录，不混成一套 |

`docs/environment/` 下的 `env_freeze_20261108.txt` 与 `before_tf_upgrade_20261108_105710.txt` 是 **Transformers 升级前**的记录，不能原样当当前锁文件。`flash_attn @ file:///tmp/fa/...` 是安装来源；原 wheel 目录已清理不等于已安装模块失效。

根 `requirements.txt` 的 `transformers<4.48` 与当前 LLaMA-Factory 的 `>=4.49.0` 冲突，不能直接同时照抄两份清单。当前选 4.49.0，并用真实 SFT、短跑与回归测试确认覆盖路径；不据此承诺所有可选模块兼容。

### 3.2 已有环境只读验收

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tempr1
cd "$HOME/Temp-R1"
unset DISABLE_VERSION_CHECK
python - <<'PY'
import importlib.metadata as metadata
import sys
import torch
from llamafactory.train.tuner import run_exp
for name in ("torch", "vllm", "transformers", "tokenizers", "flash-attn", "deepspeed", "datasets", "numpy"):
    print(name, metadata.version(name))
print("python", sys.version)
print("CUDA runtime", torch.version.cuda, "available", torch.cuda.is_available())
print("SFT_IMPORT_OK")
PY
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `source .../conda.sh` | 为未 init 的 shell 加载 conda |
| 2 | `conda activate tempr1` | 统一训练环境 |
| 3 | `cd "$HOME/Temp-R1"` | 在项目根检查 |
| 4 | `unset DISABLE_VERSION_CHECK` | 不用跳过检查冒充兼容修复 |
| 5 | Python heredoc | 打印版本、CUDA 可见性和 SFT 导入，不加载模型训练 |

导入、真实生成、SFT 前反向、GRPO 优化器更新与保存是不同验收层级。当前已有阶段证据，不为更新手册重复占 GPU。

### 3.3 新机器缺少训练环境时的安装参考

仅在没有可复用 `tempr1` 时执行。本节前置为 Linux x86_64、CUDA 驱动与 `$HOME/miniconda3` 已就位；Miniconda 缺失先按其官方平台安装流程完成。以下按既有组合整理，**未在全新机器重新完成安装验收**。

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create -n tempr1 python=3.9 -y
conda activate tempr1
cd "$HOME/Temp-R1"
python -m pip install 'torch==2.4.0' --index-url https://download.pytorch.org/whl/cu121
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `source .../conda.sh` | 加载已有 Miniconda |
| 2 | `conda create ...` | 仅为新机器创建训练环境 |
| 3 | `conda activate tempr1` | 明确后续安装目标 |
| 4 | `cd "$HOME/Temp-R1"` | 先取代码再安装项目依赖 |
| 5 | `python -m pip install torch...` | 安装当前适配使用的 CUDA wheel |

FlashAttention wheel 必须匹配 Python、Torch、CUDA/ABI、平台。本次使用 cp39 / torch2.4 / ABI FALSE 的预编译 wheel：

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)'
FA_WHEEL="$HOME/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"
curl -fL --retry 3 -o "$FA_WHEEL" 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl'
printf '%s  %s\n' 'dcbc43553821df03e83eb4bc0778054d72f0bccaa26270565b09f4ca0b680576' "$FA_WHEEL" | sha256sum -c - && python -m pip install --no-deps "$FA_WHEEL"
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `python -c 'import torch; ...'` | 确认 torch2.4 / cu121 / ABI FALSE 条件；不符就停止 |
| 2 | `FA_WHEEL=...` | 保留完整合法 wheel 文件名 |
| 3 | `curl -fL ...` | 从作者 release 下载，不把错误页当 wheel |
| 4 | `sha256sum -c - && ... pip install` | 核对本次历史 wheel SHA256 后安装；失败不继续 |

安装后可决定保留该明确下载文件；普通续跑无需再次下载。预编译 wheel 安装不等于 DeepSpeed 所有扩展都已编译。需要 CUDA toolkit 时，必须确认安装目标环境，不能在创建环境后未激活就装入其他环境。

从项目根安装，明确关键版本，避开根清单旧 Transformers 上限：

```bash
python -m pip install 'torch==2.4.0' 'torchvision==0.19.0' 'vllm==0.6.3' 'transformers==4.49.0' 'tokenizers==0.21.1' 'numpy==1.26.4' 'datasets==4.0.0' 'accelerate==1.10.1' 'tensordict==0.5.0' 'deepspeed==0.15.4' 'ray==2.51.2' 'safetensors==0.5.3' codetiming dill hydra-core pybind11 wandb IPython tensorboard -e './LLaMA-Factory[metrics]'
python -m pip check
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `python -m pip install ... -e './LLaMA-Factory[metrics]'` | 同次求解约束关键依赖，并安装仓库内 LF；不对不存在 setup 的根包 editable 安装 |
| 2 | `python -m pip check` | 逐条核实元数据冲突和缺包，不能一概忽略 |

新安装失败时保留错误并处理真实依赖，不用 `DISABLE_VERSION_CHECK=1` 掩盖问题。安装后还需导入、服务、短跑和保存验收；静态通过不保证正式 GRPO 显存足够。

### 3.4 现有检索环境

```bash
"$HOME/miniconda3/envs/retriever/bin/python" -c 'import sys, torch, faiss, transformers, fastapi, pydantic; print(sys.version); print(torch.__version__, faiss.__version__, transformers.__version__); print(fastapi.__version__, pydantic.__version__)'
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `retriever/bin/python -c ...` | 核对服务实际解释器，不修改训练环境 |

当前检索入口写死这个解释器。新机器没有该环境时，先决定复建还是适配到已有环境并验收；本轮没有验证所有功能合入一个全新环境的安装方案，不默认迁移服务。

### 本节验收

| 检查命令/检查项 | 判据 |
|---|---|
| 版本与 `SFT_IMPORT_OK` | 导入通过、关键版本与当前适配一致 |
| 新安装的 pip check | 无未经解释的冲突或缺包 |
| 检索绝对解释器 | 可导入当前服务模块 |
| 验证边界 | SFT 与短跑已通过；正式训练资源问题仍未解决 |

## 4. MultiTQ 问题数据与 parquet

### 4.1 已有部署只读检查

```bash
cd "$HOME/Temp-R1"
python - <<'PY'
from pathlib import Path
import json
import pyarrow.parquet as pq
root = Path.cwd()
for name in ("train.json", "train_MULTITQ.json", "dev.json", "test.json"):
    path = root / "data/MultiTQ" / name
    if path.is_file():
        with path.open(encoding="utf-8") as source:
            print(name, len(json.load(source)))
    else:
        print(name, "MISSING")
expected = {"train_all": 386787, "train_single": 283482, "train_multiple": 103305,
            "test_all": 54584, "test_single": 38864, "test_multiple": 15720}
for name, wanted in expected.items():
    path = root / "data/multitq_search2" / (name + ".parquet")
    rows = pq.ParquetFile(path).metadata.num_rows
    print(path.name, rows)
    assert rows == wanted, (path, rows, wanted)
PY
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/Temp-R1"` | 从项目根检查 |
| 2 | Python heredoc | 读原题条数与 parquet 元数据，不重建、不占 GPU |

| parquet | 行数 | 用途 |
|---|---|---|
| `train_all.parquet` | 386787 | Stage2 训练池 |
| `train_single.parquet` | 283482 | 单跳子集 |
| `train_multiple.parquet` | 103305 | Stage1 训练池 |
| `test_all.parquet` | 54584 | 统一评测候选池 |
| `test_single.parquet` | 38864 | 单跳评测 |
| `test_multiple.parquet` | 15720 | Stage1 验证候选池 |

原题 train/dev/test 分别为 386787 / 57979 / 54584。当前 Stage1 从 test_multiple 抽 24 题验证，不是全量测试。

### 4.2 缺失原题时恢复

完整训练集和 parquet 被忽略，克隆后不会自动获得。来源为 [MultiTQ 官方数据](https://github.com/czy1999/MultiTQ/blob/main/data/Dataset.zip)。已有 dev/test 先比较版本，不无条件覆盖。

```bash
MULTITQ_TMP=$(mktemp -d /tmp/tempr1-multitq.XXXXXX)
curl -fL --retry 3 -o "$MULTITQ_TMP/Dataset.zip" 'https://github.com/czy1999/MultiTQ/raw/main/data/Dataset.zip'
unzip -q "$MULTITQ_TMP/Dataset.zip" -d "$MULTITQ_TMP"
find "$MULTITQ_TMP" -type f -path '*/questions/*.json' -printf '%s %p\n'
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `MULTITQ_TMP=$(mktemp -d ...)` | 创建独立下载目录，避免复用旧解压结果 |
| 2 | `curl -fL ...` | 下载官方归档；失败不继续 |
| 3 | `unzip ...` | 解到独立目录，不覆盖项目题目 |
| 4 | `find ...` | 确认实际解压路径和三个文件 |

历史解压路径为 `$MULTITQ_TMP/MultiTQ/questions/`，确认位置后仅补缺失文件：

```bash
cp -n -- "$MULTITQ_TMP/MultiTQ/questions/train.json" "$HOME/Temp-R1/data/MultiTQ/train.json"
cp -n -- "$MULTITQ_TMP/MultiTQ/questions/dev.json" "$HOME/Temp-R1/data/MultiTQ/dev.json"
cp -n -- "$MULTITQ_TMP/MultiTQ/questions/test.json" "$HOME/Temp-R1/data/MultiTQ/test.json"
cmp -- "$MULTITQ_TMP/MultiTQ/questions/dev.json" "$HOME/Temp-R1/data/MultiTQ/dev.json"
cmp -- "$MULTITQ_TMP/MultiTQ/questions/test.json" "$HOME/Temp-R1/data/MultiTQ/test.json"
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cp -n ... train.json` | 只补缺失训练题目 |
| 2 | `cp -n ... dev.json` | 只补缺失开发集 |
| 3 | `cp -n ... test.json` | 只补缺失测试集 |
| 4 | `cmp ... dev.json` | 验证开发集字节一致，差异先处理 |
| 5 | `cmp ... test.json` | 验证测试版本，不混用数据 |

`$MULTITQ_TMP` 是本次下载目录，成功后可按该明确路径另行清理，不把通配删除夹在普通运行中。

### 4.3 整套 parquet 缺失时重建

先验收原题。预处理硬编码读取 `train_MULTITQ.json`，现有副本一致则复用。脚本会写六份 parquet，下面仅在整个输出目录不存在时运行。部分文件缺失时先停止，另选新目录重建、验收后再审阅消费路径切换，不能直接覆盖剩余正确文件。

```bash
cd "$HOME/Temp-R1"
cp -n -- data/MultiTQ/train.json data/MultiTQ/train_MULTITQ.json
cmp -- data/MultiTQ/train.json data/MultiTQ/train_MULTITQ.json && \
    test ! -e data/multitq_search2 && \
    PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python scripts/data_process/multitq_search.py --input_dir data/MultiTQ --local_dir data/multitq_search2 --template_type base
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/Temp-R1"` | 固定相对路径 |
| 2 | `cp -n ... train_MULTITQ.json` | 仅补缺失的硬编码文件名 |
| 3 | `cmp ... && test ! -e ... && ... python ...` | 副本一致且整套目录不存在才按 base 模板生成；任一条件失败均不写 parquet |

题型划分和教师候选只在重新准备教师数据时使用，已有输入跳过；候选脚本会覆盖同名 JSON 和统计文件，因此增加目标保护：

```bash
python scripts/data_process/qtype_classify.py --input data/MultiTQ/train.json --output_dir data/MultiTQ/split_by_qtype
test ! -e data/MultiTQ/oversampled_for_sft.json && test ! -e data/MultiTQ/oversampled_for_sft.stats.json && python scripts/data_process/create_oversampled_dataset.py
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `qtype_classify.py ...` | 按六种 qtype 写分类目录 |
| 2 | `test ! -e ... && ... create_oversampled_dataset.py` | 目标均不存在才按 seed 42 生成 1885 条候选，保护已有输入；不等于正确轨迹数 |

### 本节验收

| 检查命令/检查项 | 判据 |
|---|---|
| 原题、dev/test 版本 | 条数符合基线，版本差异已解释 |
| 六份 parquet | 全部存在，行数正确 |
| single + multiple | 分别等于对应 train/test 的 all |
| 模板与列 | 保留 base 动作提示词与 reward 所需字段 |

## 5. 时序 KG 转换与索引一致性

### 5.1 当前语料契约

规范入口为 `scripts/data_process/convert_kg_to_jsonl.py`，旧 `scripts/convert_kg_to_jsonl.py` 已删除。默认输入输出根据项目定位，不依赖调用 cwd，不再使用 `/corpus` 绝对路径。

`corpus/full.txt` 为 TAB 四列：主体、关系、客体、日期。例如：

```text
Peter_Pace<TAB>Make_a_visit<TAB>China<TAB>2007-03-22
```

对应 JSONL：

```json
{"id":"101702","contents":"Peter_Pace Make_a_visit China on 2007-03-22","subject":"Peter_Pace","predicate":"Make_a_visit","object":"China","date":"2007-03-22"}
```

六字段均为字符串。ID 从 `"0"` 起，保留原始顺序、重复事实、下划线与日期；第 101703 行对应 `"101702"`。空行、错列、空字段、非法 UTF-8、控制字符、无效日期均报错，不静默跳过。

### 5.2 现有语料只读全量核对

```bash
cd "$HOME/Temp-R1"
python scripts/data_process/convert_kg_to_jsonl.py --verify-against corpus/full_kg.jsonl
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/Temp-R1"` | 定位显式参考文件 |
| 2 | `convert_kg_to_jsonl.py --verify-against ...` | 逐行比较六字段、类型、数量与顺序，不写语料/索引、不占 GPU |

本次现场结果：

```text
VERIFIED records=461329 delimiter=tab
input_sha256=e5cfd92e9392728f45f8584442aa6f0abf916d253490f280782b9f602f4998a1
ordered_contents_sha256=1bce7e0acbbd7bf2d058a065017a8fd4264d7267ea0d09e29842da9d778c8f53
```

第二个摘要按顺序、带长度前缀编码 contents，不等于对整个 JSONL 文件直接 sha256sum。本次修复未重写既有语料、embedding 与 FAISS。

### 5.3 需要新语料时才转换

```bash
python scripts/data_process/convert_kg_to_jsonl.py --input corpus/full.txt --output corpus/full_kg.candidate.jsonl --delimiter tab
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `convert_kg_to_jsonl.py ... --output ...candidate.jsonl` | 完整成功后发布新文件，目标必须不存在；已有目标拒绝覆盖 |

支持 auto/tab/pipe/whitespace 四列模式；auto 锁定首行分隔符，不为坏行切换另一模式。新语料未验收不能替换活动文件。

FAISS 返回语料**物理行号**，不按 id 字段查字典。行数、顺序、contents 或编码配置变化后，在新目录重建配套索引；仅行数相同不能证明兼容。

### 5.4 索引验收与缺失重建

先检查既有索引：

```bash
"$HOME/miniconda3/envs/retriever/bin/python" - <<'PY'
from pathlib import Path
import faiss
root = Path.home() / "Temp-R1"
index = faiss.read_index(str(root / "corpus/kg_index/e5_Flat.index"))
with (root / "corpus/full_kg.jsonl").open("rb") as source:
    rows = sum(1 for _ in source)
print("index", index.ntotal, index.d, "corpus", rows)
assert index.ntotal == rows == 461329
assert index.d == 768
PY
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | 检索解释器 Python heredoc | CPU 读索引元数据和语料行数，不重复分配 GPU 索引 |

当前配置为 `intfloat/e5-base-v2`、mean pooling、归一化、编码长度 256、Flat、768 维。编码输入为 contents，E5 文档侧带 `passage:` 前缀。索引与 embedding 各约 1.4 GB，目录约 2.7 GB；大小仅作辅助线索。

只有缺失或确需重建且 GPU 空闲时，使用独立目录。真实参数为 `--corpus_path`，不能用旧手册的 `--corpus_file`。

```bash
cd "$HOME/Temp-R1"
INDEX_DIR="$PWD/corpus/kg_index-rebuild-$(date +%Y%m%d-%H%M%S)"
mkdir -- "$INDEX_DIR" && \
    HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 "$HOME/miniconda3/envs/retriever/bin/python" search_r1/search/index_builder.py --retrieval_method e5 --model_path intfloat/e5-base-v2 --corpus_path "$PWD/corpus/full_kg.jsonl" --save_dir "$INDEX_DIR" --use_fp16 --max_length 256 --batch_size 512 --pooling_method mean --faiss_type Flat --save_embedding
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/Temp-R1"` | 固定输入与入口位置 |
| 2 | `INDEX_DIR=...` | 选新目录，保护活动 kg_index |
| 3 | `mkdir ... && ... index_builder.py ...` | 新目录创建成功才 GPU 编码与保存 embedding、CPU 建 FAISS；失败不继续写入 |

`--use_fp16`、`--save_embedding`、`--faiss_gpu` 为 store_true，只写开关，不附 True。当前命令未传 faiss_gpu，但编码器仍使用 CUDA。复用 `--embedding_path` 时源码只核对形状，没有语料/顺序/模型指纹，不能仅以 memmap 可读证明兼容。

`build_index.sh` 会写当前 `corpus/kg_index`，正常续跑不执行。新目录完成后先验收元数据与功能，再审阅服务切换；本次文档更新不切换活动服务。

### 本节验收

| 检查命令/检查项 | 判据 |
|---|---|
| 只读 verify | 461329 条，六字段与顺序完全一致 |
| 索引 | ntotal 461329、d 768，配套语料与编码模型清楚 |
| 新语料 / 索引 | 独立路径、未覆盖活动文件，验收后才决定切换 |

## 6. 检索服务与接口验收

### 6.1 先检查已运行服务

```bash
cd "$HOME/Temp-R1"
pgrep -af -u "$(id -u)" '[r]etrieval_server.py'
tail -n 30 logs/retrieval.log
curl -sS --max-time 30 -w '\nHTTP_STATUS=%{http_code}\n' http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' -d '{"queries":["China"],"topk":1}'
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/Temp-R1"` | 定位工作根 |
| 2 | `pgrep ...` | 查当前用户服务，不固定历史 PID |
| 3 | `tail ...` | 查看最新启动段和后续错误 |
| 4 | `curl ...` | 验证默认不带分数请求的 HTTP 与结构 |

入口 `scripts/retrieval/retrieval_launch.sh` 固定 cwd、GPU 0、既有 retriever 解释器、`corpus/kg_index/e5_Flat.index`、`corpus/full_kg.jsonl`、topk 30、8000 端口。

最近同步日志显示旧进程正常退出、新进程启动，之后四个正常请求 200、非法 topk 422。旧启动段的解包异常属于修复前记录，不因整份日志还含旧 traceback 就判断新服务失败。未实现 GET /metrics 的 404 不作为检索故障判据。

### 6.2 未运行且依赖就绪时启动

健康服务跳过本节，不在同端口重复启动，不为更新手册重启。

```bash
cd "$HOME/Temp-R1"
mkdir -p logs
PYTHONUNBUFFERED=1 nohup bash scripts/retrieval/retrieval_launch.sh >> logs/retrieval.log 2>&1 < /dev/null &
tail -n 30 logs/retrieval.log
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/Temp-R1"` | 固定路径 |
| 2 | `mkdir -p logs` | 准备日志目录 |
| 3 | `nohup bash ... >> ... &` | 后台常驻，追加而不截断旧日志 |
| 4 | `tail ...` | 查看加载与 Uvicorn 启动，随后做功能验收 |

首次加载可能要等待。连接被拒绝先查最新日志和进程，确认正在启动才重试。停止时先核实自身 PID、命令与端口，再定向 TERM；不对全部 Python/Ray/vLLM 广泛杀进程。

### 6.3 五项接口合同

```bash
curl -sS --max-time 30 -w '\nHTTP_STATUS=%{http_code}\n' http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' -d '{"queries":["China"],"topk":1}'
curl -sS --max-time 30 -w '\nHTTP_STATUS=%{http_code}\n' http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' -d '{"queries":["China"],"topk":1,"return_scores":true}'
curl -sS --max-time 30 -w '\nHTTP_STATUS=%{http_code}\n' http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' -d '{"queries":["China","Japan"],"topk":1}'
curl -sS --max-time 30 -w '\nHTTP_STATUS=%{http_code}\n' http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' -d '{"queries":[],"topk":1}'
curl -sS --max-time 30 -w '\nHTTP_STATUS=%{http_code}\n' http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' -d '{"queries":["China"],"topk":0}'
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | 默认 return_scores | 验证 false 返回文档列表 |
| 2 | return_scores=true | 验证 document/score 包装 |
| 3 | 两条查询 | 验证每条查询对应一个结果组 |
| 4 | 空查询列表 | 验证空 result |
| 5 | topk=0 | 验证正整数参数约束 |

| 情况 | 本次现场结果 | 结构判据 |
|---|---|---|
| 默认 false | 200 | `result=[[document]]`，完整保留文档字段 |
| scores true | 200 | `result=[[{"document":...,"score":...}]]` |
| China / Japan | 200 | 两个结果组 |
| 空 queries | 200 | `{"result":[]}` |
| topk 0 | 422，greater_than | 明确拒绝非法参数 |

China 本次 top1 为 `Peter_Pace Make_a_visit China on 2007-03-22`，ID 101702，分数约 0.8464084。API 的 date 可经数据集解析序列化为 `2007-03-22T00:00:00`，原 JSONL 仍为 `2007-03-22`。

服务将过大 topk 限制到索引大小，拒绝负行号和 FAISS 非法哨兵。空索引或索引/语料数量不匹配会在 GPU/编码器初始化前报错。

`retrieval_launch_timelinekgqa.sh` 属于另一套语料、索引和端口，本轮没有完整建库/服务验收，不混入已验证的 MultiTQ 主线。

### 本节验收

| 检查命令/检查项 | 判据 |
|---|---|
| 最新启动段 | 完成加载，后续对应请求无异常 |
| 五项请求 | 200、200、200、200、422，结构符合约定 |
| 服务实例 | 只运行所需实例，不释放他人进程 |

## 7. 教师轨迹、筛选与 SFT 数据

### 7.1 本次已有输入

| 文件 | 用途 | 处理方式 |
|---|---|---|
| `outputs/train_sft_oversampled_merged.jsonl` | 作者 MultiTQ 合并轨迹，1055 条 | 保留，固定版本记录的 valid / answer_correct 标记均 true |
| `outputs/gpt4o-1000-cron-new.jsonl` | 另一套作者轨迹 | 保留，不当成当前 MultiTQ 必需重新生成步骤 |
| `LLaMA-Factory/data/multitq_sft_train.json` | 当前 SFT ShareGPT 数据 | 可读则复用 |
| `LLaMA-Factory/data/dataset_info.json` | 注册和角色映射 | 注册名 multitq_sft_train |

存储的正确标记是作者产物元数据，不等于本轮独立重新判卷。当前 LF 数据为 ShareGPT，消息字段 conversations，角色 system/human/gpt/observation；不能写成旧手册的 Alpaca 三字段格式。

```bash
cd "$HOME/Temp-R1"
python - <<'PY'
from pathlib import Path
import json
root = Path.cwd()
with (root / "outputs/train_sft_oversampled_merged.jsonl").open(encoding="utf-8") as source:
    rows = [json.loads(line) for line in source]
print("teacher trajectories", len(rows))
with (root / "LLaMA-Factory/data/multitq_sft_train.json").open(encoding="utf-8") as source:
    records = json.load(source)
assert len(records) >= 1000
assert all(isinstance(row.get("conversations"), list) for row in records)
with (root / "LLaMA-Factory/data/dataset_info.json").open(encoding="utf-8") as source:
    info = json.load(source)["multitq_sft_train"]
print("SFT rows", len(records), "registration", info)
assert info["file_name"] == "multitq_sft_train.json"
assert info["formatting"] == "sharegpt"
PY
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/Temp-R1"` | 定位当前输入 |
| 2 | Python heredoc | 检查轨迹、SFT 消息格式与注册，不调用教师 API |

`max_samples: 1000` 是训练上限，不应把作者输入强行裁成恰好 1000 条。

### 7.2 只有格式数据缺失时转换

`llama-factory-dataset.py` 没有输入输出 CLI，直接执行 main 会访问写死的 `/outputs/...` 和 `/LLaMA-Factory/...`。已有正确数据跳过；缺失时调用其转换函数并传真实项目路径：

```bash
cd "$HOME/Temp-R1"
python - <<'PY'
from pathlib import Path
import os
import runpy
import tempfile
root = Path.cwd()
source = root / "outputs/train_sft_oversampled_merged.jsonl"
target = root / "LLaMA-Factory/data/multitq_sft_train.json"
if target.exists() or target.is_symlink():
    raise SystemExit("目标已存在，保留并停止；先做只读验收")
namespace = runpy.run_path(str(root / "scripts/data_process/llama-factory-dataset.py"), run_name="manual_conversion")
with tempfile.TemporaryDirectory(prefix="tempr1-sft-", dir=str(target.parent)) as temporary:
    candidate = Path(temporary) / target.name
    namespace["convert_to_llama_factory_format"](str(source), str(candidate))
    os.link(str(candidate), str(target))
PY
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/Temp-R1"` | 固定项目根 |
| 2 | Python heredoc / runpy | 避开硬编码 main，在目标盘临时转换、无覆盖发布；已有文件、链接或竞争创建均不覆盖，临时目录自动清理 |

转换器插入固定 system 提示词与 human 问题，按 information 段划分 gpt 输出和 observation，函数本身不筛正确性。数据注册若缺失需按当前 ShareGPT 映射补充，不覆盖整份 dataset_info。

### 7.3 可选重新生成教师轨迹的限制

当前不需要 API key，也不重新付费。将来独立生成前需处理以下事项，不能直接执行旧包装器：

| 项目 | 固定源码事实 | 处理要求 |
|---|---|---|
| 答案字段 | 当前 1885 条候选含 answers，生成器读取 example['answer'] | 先审阅并测试兼容修复；直接执行可能 KeyError |
| 路径 | 包装器含 /data、/outputs 绝对路径 | 显式使用当前路径和新的输出文件 |
| 凭据 | 包装器未可靠传入已注释 API 配置 | 单独核对参数，不把 key 写进源码、日志或仓库 |
| 教师后端 | model_api 可为 openai / local，默认 model 为 gpt-4o | 默认参数不证明合并数据的完整教师来源 |
| 数量 | num_samples 只切输入片段 | 过滤后可能少于该数量 |
| 筛选 | filter_correct_only 为 store_true | 开启才只保留 valid 且正确轨迹，不附 True |
| 覆盖 | 原入口可能写作者合并文件 | 新路径输出，不覆盖当前训练输入 |

格式检查要求早期 plan、四个规划标记、answer 和工具标签数量闭合；正确性是归一化预测片段命中某个标准答案，不等同完整多答案集合 EM。没有强制每条都用 search/filter/rank。

换教师、改筛选或采样均属实验配置变化，需另记来源。本轮只维护手册，未修复可选生成分支，也未调用付费 API；当前复用作者数据的主线不受该分支阻塞。

### 本节验收

| 检查命令/检查项 | 判据 |
|---|---|
| 输入和注册 | JSON 可读，SFT 至少 1000 条、ShareGPT 角色映射正确 |
| 当前路线 | 跳过付费生成与重复转换 |
| 新生成路线 | 字段、路径、筛选、凭据和独立输出先验收 |

## 8. SFT Cold Start：已完成的 1.5B 阶段

### 8.1 配置与完成证据

正式配置为 `LLaMA-Factory/examples/train_full/tempr1_qwen_1p5b_sft.yaml`，入口为 `scripts/multitq/train_sft_qwen_1p5b.sh`。不再仅修改作者原 YAML 的两行就假定配置已适配。

| 项目 | 当前值 |
|---|---|
| 模型 / 模板 | Qwen/Qwen2.5-1.5B-Instruct / qwen |
| 训练类型 | full SFT，非 LoRA |
| 数据 / 上限 | multitq_sft_train / 1000 |
| 长度 | 8192 |
| 单卡 micro / 梯度累积 | 1 / 20，有效 batch 20 |
| 学习率 / 轮数 | 2e-5 / 2 epoch |
| 调度 / warmup | cosine / 0.1 |
| 精度与内存 | bf16、FA2、gradient checkpointing、ZeRO-3 CPU offload |
| 保存 | 每 epoch；overwrite_output_dir=false |
| 正式输出 | LLaMA-Factory/saves/qwen2.5-1.5b-it/full/sft/ |
| 日志 | logs/sft_qwen_1p5b.log |

日志记录 1000 样本、100 次优化更新、2.0 epoch、全程平均 `train_loss=0.14289076924324035`、`train_runtime=2998.6568` 秒与最终目录保存成功。末次分段 loss 不等于全程平均，更不等于问答准确率。

### 8.2 最终权重与 tokenizer 只读检查

```bash
cd "$HOME/Temp-R1"
python - <<'PY'
from pathlib import Path
import json
from safetensors import safe_open
from transformers import AutoTokenizer
path = Path("LLaMA-Factory/saves/qwen2.5-1.5b-it/full/sft")
with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as weights:
    count = len(weights.keys())
print("SFT tensor count", count)
assert count == 338
with (path / "config.json").open(encoding="utf-8") as source:
    config = json.load(source)
assert config["model_type"] == "qwen2"
tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
text = tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}], tokenize=False, add_generation_prompt=True)
assert isinstance(text, str) and text
print("SFT_TOKENIZER_OK")
PY
grep -E 'train_loss|train_runtime|train_steps_per_second' logs/sft_qwen_1p5b.log | tail -n 8
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/Temp-R1"` | 定位正式输出 |
| 2 | Python heredoc | CPU 检查张量键、配置和 tokenizer 模板，不启动训练 |
| 3 | `grep ...` | 对照正式完成指标 |

现场已确认 338 张量与模板可读，模型文件约 3.09 GB。本检查不逐张量校验所有值，也不替代质量评测。

### 8.3 新部署且无正式输出时的启动参考

当前已完成，跳过。仅整个正式输出目录不存在时启动：

```bash
cd "$HOME/Temp-R1"
test ! -e "$HOME/Temp-R1/LLaMA-Factory/saves/qwen2.5-1.5b-it/full/sft" && tmux new-session -d -s sft1p5b -c "$HOME/Temp-R1" 'bash scripts/multitq/train_sft_qwen_1p5b.sh'
tmux capture-pane -pt '=sft1p5b' -S -40
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/Temp-R1"` | 使用适配入口 |
| 2 | `test ! -e ... && tmux new-session ...` | 保护正式目录，缺失才启动单卡 SFT |
| 3 | `tmux capture-pane ...` | 查看实际启动，不以会话存在当成功 |

入口设置 FORCE_TORCHRUN=1、NPROC_PER_NODE=1，以 `python -m llamafactory.cli` 启动，内部进入 LF，因此 YAML 相对输出在 LF 下。脚本 tee 会覆盖固定 SFT 日志；重复实验应先审阅独立输出和日志配置。

旧 checkpoint-50 / checkpoint-100 及优化器状态已删除。LF 可从完整 checkpoint 恢复优化器与步数，但当前最终独立模型不能补回已删状态；它继续作为 GRPO 初始化。重新加载模型训练不等于精确续训。

### 本节验收

| 检查命令/检查项 | 判据 |
|---|---|
| 完成日志 | 100 更新、2 epoch、最终保存 |
| 权重、config、tokenizer | 338 张量与 Qwen 模板可读 |
| 下阶段路径 | 加载 full/sft，不依赖已删短跑或 checkpoint |
| 当前动作 | 跳过已完成阶段，不重训 |

## 9. GRPO Stage1：短跑通过，正式训练暂停

### 9.1 输入、参数和当前结果

正式入口为 `scripts/multitq/train_grpo_stage1_1p5b.sh`，使用 `verl.trainer.main_ppo_format`；短跑入口为同目录 `train_grpo_stage1_1p5b_smoke.sh`。

| 项目 | 正式 Stage1 | 两步短跑 |
|---|---|---|
| 初始化 | 正式 SFT | 正式 SFT |
| 训练池 | multiple 103305 条 | 从中取 8 条 |
| prompt batch / G | 60 / 5 | 2 / 5 |
| 每外层步候选轨迹 | 300 | 10 |
| PPO mini / micro | 30 / 1 | 2 / 1 |
| 外层训练步数 | 80 | 2 |
| 验证池 / 抽样 | test_multiple / 24 | test_multiple / 2 |
| 学习率 / KL | 5e-7 / 0.01 | 以短跑配置为准 |

80 指外层训练步数；每个外层步还包含多个 actor 优化器小批次，不等于总计只执行 80 次 optimizer.step。正式配置还有 grad_clip 5.0、temperature 1.0、vLLM 显存比例 0.4、TP 1、dummy_hf、actor/ref logprob micro 1、单 GPU、梯度检查点、actor 参数/梯度/优化器 offload 和 reference 参数 offload。

长度为 prompt 5200、response 1000、start 2000、observation 1400；最多 3 轮，检索 topk 30。保留 state_masking=true、no_think_rl=false 和规划/筛选/排序动作空间。micro 调整为显存适配，batch、G、长度决定真实训练协议，不能统称只改路径。

正式运行首步 loss.backward OOM：还需 2.84 GiB，现场仅余 1.29 GiB；错误同时记录其他占用约 12.82 GiB、检索约 2.86 GiB。身份不能仅凭容器 GPU 列表认定。短跑不保证完整长批次峰值足够。

**继续暂停，等待资源与用户决定，不自动启动。**

### 9.2 已应用修复与验证边界

| 类型 | 内容 | 证据与边界 |
|---|---|---|
| 版本兼容 | 训练 Transformers 4.49.0 | 导入、正式 SFT、短跑覆盖当前路径 |
| 等价显存优化 | 非 remove-padding 分支只投影响应所需 response_length+1 个 logits | CPU 输出/损失/梯度等价与两步 GRPO；完整上下文仍参与前向 |
| 停止条件修复 | 外层比较从 >= 改为 > | N 更新不再少 1；入口含幂等检查 |
| 语料与接口修复 | 六字段转换、分数开关、批查询、topk/行号边界 | 48 测试、461329 行、五项 HTTP 通过 |

vLLM 预分配不能保证独占整卡或消除反向峰值，当前实现管理阶段切换的推理缓存。增大 gpu_memory_utilization 不是已验证 OOM 修复，不以预占显存代替可用资源检查。

### 9.3 未来恢复前的检查与 tmux 方式

以下只读预检不恢复训练：

```bash
cd "$HOME/Temp-R1"
nvidia-smi
pgrep -af -u "$(id -u)" 'verl.trainer|llamafactory|[r]etrieval_server.py'
bash -n scripts/multitq/train_grpo_stage1_1p5b.sh
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/Temp-R1"` | 定位入口 |
| 2 | `nvidia-smi` | 查当时资源，不沿用旧剩余量 |
| 3 | `pgrep ...` | 查已有自身任务，避免重复训练 |
| 4 | `bash -n ...` | 只解析，不执行训练或入口内补丁 |

下面仅供未来：用户要求恢复、SFT/parquet/检索验收通过且资源足够后使用。它创建新时间戳实验，仍从 SFT 开始，不恢复失败运行优化器。固定 GRPO 日志被 tee 覆盖；旧失败证据在本文固定 SHA 中可读，若要同机保留各次日志，先另定日志命名方案。

```bash
tmux new-session -d -s grpo-stage1 -c "$HOME/Temp-R1" 'bash scripts/multitq/train_grpo_stage1_1p5b.sh'
tmux capture-pane -pt '=grpo-stage1' -S -60
tail -n 40 "$HOME/Temp-R1/logs/grpo_stage1_1p5b.log"
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `tmux new-session ...` | 用户决定后持久启动新的 Stage1 |
| 2 | `tmux capture-pane ...` | 查看初始化、rollout 与实际更新 |
| 3 | `tail ...` | 查最新指标、进度与异常 |

GPU 忙碌、进程存在、出现 loss 或完成 rollout 都不能单独证明参数更新成功。需要更新指标、后续步数、保存与最终退出相互印证。

### 9.4 保存与恢复边界

| 内容 | 路径 / 行为 |
|---|---|
| Hydra | logs/grpo_runs/实验名/ |
| 权重根 | verl_checkpoints/实验名/ |
| 每 10 步保存 | actor/global_step_10 … actor/global_step_80 |
| 验证预测 | 保存根中的 val_preds_step_N.json |
| 当前保存 | HF 模型和 tokenizer |
| 没有完整保存 | 优化器、RNG、dataloader 等精确恢复所需状态 |

这个 fork 没有读取 resume_step 恢复优化器或计数，每次 fit 从初始计数开始。`+trainer.resume_step=40` 不是真正续训，加载 global_step_N 是权重初始化。

修复后更新为 1…80，结束还会验证一次，日志计数可能为 81，不代表第 81 次参数更新。中断不保证额外保存。需检查第 80 步可读产物与完成记录，不能只看目录名。

### 本节验收

| 检查命令/检查项 | 判据 |
|---|---|
| 当前状态 | 首步 OOM 后暂停，无 80 步完成证据 |
| 未来启动 | 资源和输入现场复查，用户决定后执行 |
| 正式进度 | 更新指标与步数推进，不只是 rollout |
| 正式完成 | 更新 1…80、结束验证/退出、第 80 步权重可读 |
| 恢复表述 | 权重初始化，不承诺恢复 GRPO 优化器 |

## 10. Stage2 与统一评测：待适配部分

当前没有已验收的单 A6000 Stage2 或完整评测入口。下表为固定源码事实和待处理事项，不是可直接执行配置。

| 入口 | 源码事实 | 当前限制 |
|---|---|---|
| scripts/multitq/train_grpo_stage2.sh | all、G8、KL0.001、500 步、save/test20 | anaconda3、多 GPU、checkpoint 占位、无效 resume_step，待适配 |
| scripts/multitq/evaluate.sh | main_ppo_format、test_all 抽 500、长响应和 batch500 | 模型占位、多卡、广泛 pkill/rm、显存未验收，不直接执行 |
| scripts/timelinekgqa/ | 其他数据集流程 | 配套输入和服务不同，未纳入本次主线 |

Stage2 需加载实际可读 Stage1 权重，并记录优化器、步数和采样设置，不能靠 resume_step 或只换路径承诺续训。

### 10.1 指标口径

| 场景 | 数据范围 | 指标与规则 |
|---|---|---|
| 当前 Stage1 验证 | test_multiple、seed42 抽 24 | val/test_score/{ds} 为平均规则奖励；qa_em_format 为归一化包含匹配 |
| 作者 evaluate | test_all 抽 500 | format 分支，不能称完整测试严格 accuracy |
| 作者 Stage2 验证 | test_all 抽 30 | main_ppo / qa_em 为归一化严格相等，与 Stage1 不同 |
| 待建立正式对照 | Base / SFT / Stage1 / Stage2 | 统一划分、抽样或全量、生成参数、检索和评分后比较 |

平均规则奖励、包含匹配、严格 EM、完整多答案集合准确率不能互换。预测和指标一起保存。真正只评测需同时 val_only=true、val_before_train=true，只改前者不够。

旧手册的 0.210 / 0.374 / 0.532 及自定允许偏差未在本轮核验原论文表格、划分和评分，不作为通过标准。SFT loss、短跑或检索 200 不代表论文成绩已复现。

### 本节验收

| 检查命令/检查项 | 判据 |
|---|---|
| Stage2 | 真实权重、单卡资源、奖励、保存语义审阅通过 |
| 评测 | 无广泛清理/杀进程，路径、batch、长度与只评测开关适配 |
| 正式结论 | 同协议对照、预测与完成证据齐全；当前不作已复现声明 |

## 11. 回归测试与修复验收

`tests/` 包含 `test_kg_conversion.py` 的 34 项、`test_retrieval_contract.py` 的 14 项回归测试。它是应长期保留的源码，不是旧训练结果。

```bash
cd "$HOME/Temp-R1"
"$HOME/miniconda3/envs/tempr1/bin/python" -B -m unittest discover -s tests -p 'test_*.py' -v
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/Temp-R1"` | 使用当前源码 |
| 2 | `tempr1/bin/python -B -m unittest ...` | CPU 回归，不加载实际检索模型，不占 GPU；-B 不写 pyc |

本次 Linux 训练环境结果为 **48 项全部通过，无跳过**。检索环境单跑 14 项时曾缺 httpx 跳过 TestClient；随后训练环境全部通过，并用五项真实请求覆盖已部署服务。

覆盖格式、日期、ID、换行、原子发布、防覆盖、同文件/链接/竞态、只读核对、分数开关、批查询、空查询、topk、FAISS 哨兵和行数不匹配等。不证明仓库所有可选模块无缺陷，也不覆盖正式训练峰值或 Stage2 效果。

### 本节验收

| 检查命令/检查项 | 判据 |
|---|---|
| 测试 | 48 全通过，若 skip 则说明原因并补验收 |
| 语料 | 461329 条六字段顺序一致 |
| 实际服务 | 五项 HTTP 状态和结构正确 |
| 证据边界 | CPU 测试不等于 GPU 训练或论文效果 |

## 12. tmux 与 GitHub 单向同步

### 12.1 会话检查

```bash
tmux ls
pgrep -af -u "$(id -u)" '[a]utosync.sh'
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `tmux ls` | 查看持久会话；无训练会话不等于无产物 |
| 2 | `pgrep ...` | 核实同步脚本而非只看包装会话 |

进入用 tmux attach -t 会话名；Ctrl+b 再 d 脱离，不结束任务。会话结束可能成功也可能失败，仍看日志和权重。

### 12.2 范围与限制

服务器向 GitHub 单向推送，不自动拉回网页修改。桌面手册本次重命名也不会自动更新服务器历史文档。

| 内容 | 当前规则 |
|---|---|
| 同步入口 / 日志 | scripts/github-sync/autosync.sh / 同目录 autosync.log |
| 会话 / 远端 / 分支 | sync / backup / main |
| 间隔 / 守卫 | 60 秒；本轮暂存超 50 MiB 的改动撤出并记 skipped(big) |
| 源码、配置、测试、文档 | 正常跟踪，pyc / 安装元数据除外 |
| outputs JSONL、教师候选 | 已跟踪，属于训练输入 |
| logs/*.log | 根 ignore 明确例外，正式日志被跟踪 |
| 同步自身 autosync.log | 被忽略，避免自我增长 |
| 训练原题副本、GRPO parquet、KG 索引 | 忽略，另行恢复或保管 |
| SFT / GRPO 权重 | saves/ 与 verl_checkpoints/ 忽略，另行保管 |

已提交大文件仍可存在，50 MiB 守卫只处理本轮暂存改动，不删除工作文件、不重写历史。ignore 也不保护磁盘文件免于删除。历史同步说明部分范围已过时，以当前 .gitignore 为准。

### 12.3 已配置同步的启停与验收

仅恢复既有同步器，不重新生成密钥或重建历史。同一共享工作区只允许一节点运行同步。

```bash
tmux has-session -t '=sync' 2>/dev/null || tmux new-session -d -s sync -c "$HOME/Temp-R1" bash -c 'set -o pipefail; ROOT="$HOME/Temp-R1" REMOTE=backup BRANCH=main INTERVAL=60 MAX_MB=50 RETRIES=3 bash "$HOME/Temp-R1/scripts/github-sync/autosync.sh" 2>&1 | tee -a "$HOME/Temp-R1/scripts/github-sync/autosync.log"'
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `tmux has-session ... || tmux new-session ...` | exact sync 不存在才启动，不影响其他会话 |

同步器依次 add、撤出超限、提交与推送。NEED_PUSH 只在当前进程新提交后置 1，进程重启后已有未推送提交不一定补推，必须比较 SHA：

```bash
cd "$HOME/Temp-R1"
git rev-parse HEAD
git ls-remote backup refs/heads/main
tail -n 12 scripts/github-sync/autosync.log
git status --short
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/Temp-R1"` | 核实仓库 |
| 2 | `git rev-parse HEAD` | 本地完整 SHA |
| 3 | `git ls-remote ...` | 远端 main 第一列须与本地一致 |
| 4 | `tail ...` | 查看本次推送，不只看历史 pushed |
| 5 | `git status --short` | 识别仍有改动；空输出不表示 ignored 大文件已上传 |

需暂停同步审阅时：

```bash
tmux kill-session -t '=sync'
pgrep -af -u "$(id -u)" '[a]utosync.sh'
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `tmux kill-session -t '=sync'` | 仅停同步；无会话报错不影响检索或训练 |
| 2 | `pgrep ...` | 查看同工作区残留同步，不全局杀进程 |

分叉时先停同步、读取双方提交再审阅合并。fetch 后直接 push 不自动解决 non-fast-forward；不以 force push、orphan 或删历史作常规修复。浅克隆不等于必须重建历史。

### 本节验收

| 检查命令/检查项 | 判据 |
|---|---|
| 同步进程 | 同工作区至多一套 |
| 实时 SHA 与树 | 双方一致，预期文件在该提交中确实存在 |
| ignored 大文件 | 独立验收与保管，不以 GitHub 树判断不存在 |
| 训练状态 | 同步运行不代表训练恢复 |

## 13. 清理历史与保留范围

本节记录已执行清理，不要求再次运行历史整理工具。正常续跑不删除缓存、数据和日志，不附带全局 pkill、ray stop 或递归清理。

| 已完成事项 | 结果与边界 |
|---|---|
| 删除 SFT checkpoint-50 / 100 | 约 40.28 GiB，优化器状态已删，最终模型保留 |
| 删除旧 SFT / GRPO 短跑权重 | 约 9.53 GiB，保留源码与验证结论 |
| 模型类清理合计 | 候选大小约 49.81 GiB，实际释放受共享文件系统影响 |
| 清理旧构建/预检/短跑日志、三套 Hydra、TensorBoard 事件 | 保留正式 SFT、最近 GRPO 失败、活动检索日志 |
| 项目外备份与审计文本 | 两个明确 Temp-R1 备份和旧审计文本已处理 |
| 自身旧 pip / torchelastic / Ray / wheel / helper | 限定已结束路径，保留其他项目与活动 IPC |
| 空 verl_checkpoints | 删除，保存时自动建立 |

旧 Ray/conda/pip 的 defunct 为 Z、RSS0、PPID1，不继续占训练显存，不能靠再次 kill 清除进程表项。SSR、Miniconda、VSCode、HF 模型缓存与用途未确认的 BERT 缓存不按本项目垃圾删除。

再检查时只读列出，GitHub 无法显示 ignored 文件的实际状态：

```bash
cd "$HOME/Temp-R1"
du -h --max-depth=2 --exclude=.git . | sort -h | tail -n 35
find outputs logs LLaMA-Factory/saves -maxdepth 6 -type f -printf '%s %p\n' | sort -n
find . -path './.git' -prune -o -type d -empty -print
```

| # | 命令 | 作用 |
|---|---|---|
| 1 | `cd "$HOME/Temp-R1"` | 限定本项目 |
| 2 | `du ...` | 看实际磁盘占用 |
| 3 | `find ... -type f ...` | 列输入、日志和模型，不删除 |
| 4 | `find ... -empty -print` | 跳过 Git 列空目录，不批量删除 |

后续删除前逐个核实目标、归属、运行依赖和恢复需要。保留 tests、作者输入、最终 SFT、时序语料和配套索引、关键日志、模型缓存与 editable 安装元数据。

### 本节验收

| 检查命令/检查项 | 判据 |
|---|---|
| 结构 | 入口与日志归位，无需保留旧空输出壳 |
| 正式权重/语料/索引 | 可读、路径不变、服务功能通过 |
| 范围 | 不删除其他用户或项目内容 |
| 输入与源码 | outputs 作者输入、tests 保留 |

## 14. 常见故障与处理

| 现象 | 已确认或待验证原因 | 处置与复查 |
|---|---|---|
| make_batched_videos 导入失败 | 当前 LF 不支持旧 Transformers 4.47.1 | 现有 tempr1 使用已验证 4.49.0 / tokenizers 0.21.1 后复查，不另建 SFT 环境 |
| 升级 Torch/vLLM 后出错 | 本项目基于旧接口 | 回到有证据组合并核对源码，不整套盲升 |
| Faiss 的 _ARRAY_API 错误 | NumPy 2 与旧二进制冲突 | 核对检索解释器与 NumPy1.26.4，复查导入 |
| 转换拒绝现有输出 | 防覆盖正常工作 | 已有语料用 verify，新转换选不存在的路径 |
| 语料日期/字段/顺序不一致 | 坏输入或配套关系改变 | 查报错物理行，不跳过坏行或覆盖活动文件 |
| 索引与语料数量不符 | 文件不配套 | 查来源、模型、顺序与路径，必要时新目录建库 |
| 默认无分数 500 / 双查询错形状 | 旧服务无条件解包 | 确认进程加载修复版，做五项接口合同 |
| 200 但返回结构错误 | HTTP 存活不等于合同正确 | 核对查询组数、文档、分数和日期 |
| topk0 返回 422 | 正常参数约束 | 改正整数，不当宕机 |
| metrics404 | 未实现此路由 | 用 POST retrieve 验收，不伪造 metrics |
| SFT 非空输出拒绝重训 | 正式权重保护或 checkpoint 恢复分支 | 完成则复用；重复实验另定输出，核对完整 checkpoint 才谈续训 |
| GRPO 首步反向 OOM | 长批次峰值和共享卡占用 | 保持暂停等待资源，短跑或 vLLM 预占不保证可跑 |
| observation 截断 / KV recompute | 长度和缓存压力 | 看后续进度与异常，warning 不直接等于失败，改配置需记录 |
| resume_step 无效 | trainer 未实现此状态恢复 | 明确权重初始化，不承诺精确续训 |
| 教师生成 answer KeyError | 数据为 answers 字段 | 先修复测试；当前已有作者数据路线跳过该分支 |
| 同步会话在但无提交 | 忽略、超限、Git 错误或分叉 | 看 status、ignore、日志与双方 SHA |
| 推送权限错误 | 密钥或写权限不符 | 核对 Temp-R1 专用密钥；脚本用 -F /dev/null 避开 SSR 全局密钥 |
| GH001 / 大文件错误 | 单文件硬上限，不能一概当警告 | 停同步识别对象和历史，审阅存储/历史方案，不自动重建 main |
| /proc FD Stale file handle | 进程/共享盘状态变化 | 不把广泛扫描 proc 当清理前提，先限定范围检查 |

## 15. 证据来源、边界与下一步

| 事实 | 来源 |
|---|---|
| 固定源码 | [fd28864b 提交](https://github.com/2200170080/Temp-R1/commit/fd28864b4229287e19fb3c6c7451ce0ed224c405)，本轮重新查 main 后固定 |
| SFT 配置与入口 | [正式 YAML](https://github.com/2200170080/Temp-R1/blob/fd28864b4229287e19fb3c6c7451ce0ed224c405/LLaMA-Factory/examples/train_full/tempr1_qwen_1p5b_sft.yaml)、[入口](https://github.com/2200170080/Temp-R1/blob/fd28864b4229287e19fb3c6c7451ce0ed224c405/scripts/multitq/train_sft_qwen_1p5b.sh) |
| SFT 完成、loss、耗时 | [正式日志](https://github.com/2200170080/Temp-R1/blob/fd28864b4229287e19fb3c6c7451ce0ed224c405/logs/sft_qwen_1p5b.log) |
| GRPO 参数与 OOM | [单卡入口](https://github.com/2200170080/Temp-R1/blob/fd28864b4229287e19fb3c6c7451ce0ed224c405/scripts/multitq/train_grpo_stage1_1p5b.sh)、[失败日志](https://github.com/2200170080/Temp-R1/blob/fd28864b4229287e19fb3c6c7451ce0ed224c405/logs/grpo_stage1_1p5b.log) |
| 保存、停止、评分 | [ray_trainer](https://github.com/2200170080/Temp-R1/blob/fd28864b4229287e19fb3c6c7451ce0ed224c405/verl/trainer/ppo/ray_trainer.py)、[fsdp_workers](https://github.com/2200170080/Temp-R1/blob/fd28864b4229287e19fb3c6c7451ce0ed224c405/verl/workers/fsdp_workers.py)、main_ppo_format / main_ppo 和对应 reward |
| 转换与接口实现 | [KG 转换器](https://github.com/2200170080/Temp-R1/blob/fd28864b4229287e19fb3c6c7451ce0ed224c405/scripts/data_process/convert_kg_to_jsonl.py)、[服务](https://github.com/2200170080/Temp-R1/blob/fd28864b4229287e19fb3c6c7451ce0ed224c405/search_r1/search/retrieval_server.py)、[测试](https://github.com/2200170080/Temp-R1/tree/fd28864b4229287e19fb3c6c7451ce0ed224c405/tests) |
| 实际服务重启后结果 | [检索日志](https://github.com/2200170080/Temp-R1/blob/fd28864b4229287e19fb3c6c7451ce0ed224c405/logs/retrieval.log)与用户五项 curl 现场输出 |
| 461329 行、48 测试、权重与清理 | 本会话用户现场输出，不由 GitHub 缺 ignored 文件反推 |
| 教师输入 | 固定作者合并 JSONL 与注册；文件名不独立证明教师版本 |
| 环境 | 历史记录、后续升级、现场导入；旧 freeze 非当前完整锁文件 |
| 同步 | [同步源码](https://github.com/2200170080/Temp-R1/blob/fd28864b4229287e19fb3c6c7451ce0ed224c405/scripts/github-sync/autosync.sh)、[ignore](https://github.com/2200170080/Temp-R1/blob/fd28864b4229287e19fb3c6c7451ce0ed224c405/.gitignore) |

待后续解决：资源充足时正式 Stage1 能否完成；Stage2/评测单卡适配与评分协议；可选教师生成的字段兼容；新机器完整安装及是否迁移检索环境。本轮不为这些分支作成功保证。

下一步保持：保留正式 SFT、作者输入与检索服务，等待 GPU 资源及用户恢复指令，再核对正式 Stage1 条件。本文只在桌面更新为 `Temp-R1_Reproduction_Guide.md`，服务器 `docs/Temp-R1_部署手册.md` 和 GitHub 文件未在本轮修改。

本轮文档校验：16 章、33 个命令块及对应作用表；32 段 Bash、1 段 PowerShell 通过解析，6 段内嵌 Python 通过 3.9 语法检查。另在自动清理的临时目录验证了实际 ShareGPT 转换函数、角色映射、已有输出保护与竞争创建保护。没有执行手册中的远程安装、服务启动、训练或清理；静态通过不等于新机器部署已经验证。
