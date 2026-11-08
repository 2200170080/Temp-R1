# Temp-R1 环境部署手册

> 适用对象：单卡 A6000（48 GB）服务器上从零复现 Temp-R1（Qwen2.5-1.5B）。
> 事实基线：2026-11-08 实测，机器 `ecs-instance-cdb4917c`，运行根 `/root/user/cs_tcci_zengdexin/Temp-R1`。
> 凡标「未核对」的条目，是本手册写作时尚未在服务器上验证过的，执行前必须先确认。

## 0. 环境约定与全流程总览

### 0.1 机器事实基线（2026-11-08 实测）

| 项 | 值 | 取数方式 |
|---|---|---|
| GPU | 1 × NVIDIA RTX A6000，49140 MiB，驱动 550.54.15，CUDA 12.4 | `nvidia-smi` |
| CPU / 内存 | 128 核 / 755 GB（可用 590 GB） | `nproc`、`free -g` |
| 根盘 | `/` overlay 457 G（清理后可用约 190 G） | `df -h /` |
| 家目录 | `/root` 为 Ceph `172.30.20.24:6789:/ltaidc`，665 T | `df -h ~` |
| 共享内存 | `/dev/shm` 90 G | `df -h /dev/shm` |
| conda | `~/miniconda3`，conda 26.9.4，**未 init** | `ls ~/miniconda3` |
| 运行根 | `/root/user/cs_tcci_zengdexin/Temp-R1`，978 M | `du -sh ~/Temp-R1` |
| 宿主机名 | `ecs-instance-cdb4917c`（另一台 `ecs-instance-e1330fb1` 的卡长期被他人占用，不要用） | `hostname` |

### 0.2 全流程一览

| 章 | 做什么 | 产物 | 预计耗时 |
|---|---|---|---|
| 1 | 登录、确认机器与磁盘 | 无 | 2 min |
| 2 | 建 `tempr1` 环境（torch / vllm / flash-attn / LLaMA-Factory） | conda 环境，约 15 G | 60–90 min |
| 3 | 拉代码 | `~/Temp-R1` 978 M | 3 min |
| 4 | 下载 MultiTQ、切题型、过采样、生成 parquet | `data/multitq_search2/*.parquet` 499 M | 10 min |
| 5 | 建检索环境、建索引、起检索服务 | `corpus/kg_index/` 2.8 G，服务 :8000 | 60 min（含编码 46 万条） |
| 6 | 脚本路径适配（三处 sed + 两处重写） | 可执行的训练脚本 | 10 min |
| 7 | 生成 1,000 条 SFT 轨迹 | `outputs/train_sft_oversampled_merged.jsonl` | 1–2 h（需 API key） |
| 8 | SFT 训练 1.5B | `LLaMA-Factory` 输出 checkpoint | 约 2 h |
| 9 | GRPO stage1 + stage2 | `verl_checkpoints/` | 12–20 h |
| 10 | 评测 Base / SFT / GRPO 三档 | 三组 accuracy | 每档 20–40 min |
| 11 | 常见错误对照 | 无 | 按需 |
| 12 | tmux 速查与换机器 | 无 | 按需 |

### 0.3 三条贯穿全程的约定

| # | 约定 | 原因 |
|---|---|---|
| 1 | 每条命令**单独一行**，不用 `\` 续行拼 conda 命令 | 反斜杠续行会让 conda 把参数当成多余项，报 `unrecognized arguments` |
| 2 | 新开 shell 先 `source ~/miniconda3/etc/profile.d/conda.sh` | conda 未 init，否则 `pip`/`conda` 直接 `command not found` |
| 3 | 只改**路径与并行方式**，不改超参 | 复现要求原配置；改 G / β / lr 会让结果不可比 |

### 本节验收

| 检查命令 | 期望输出 |
|---|---|
| `nvidia-smi --query-gpu=name,memory.total --format=csv` | `NVIDIA RTX A6000, 49140 MiB` |
| `df -h / \| tail -1` | 可用空间 ≥ 100 G |
| `ls -d ~/miniconda3` | 目录存在 |

## 1. 登录与目录准备

```bash
ssh cs_tcci_zengdexin@<host>
hostname
nvidia-smi
df -h / ; df -h ~ ; nproc ; free -g | head -2
```
| # | 命令 | 作用 |
|---|---|---|
| 1 | `ssh cs_tcci_zengdexin@<host>` | 登录目标机器，用户名与运行根绑定 |
| 2 | `hostname` | 确认落在 `ecs-instance-cdb4917c`；换到 `e1330fb1` 会因卡被占用而白跑 |
| 3 | `nvidia-smi` | 训练前必看：确认显存空闲、没有他人进程 |
| 4 | `df -h / ; df -h ~` | 根盘是 overlay、家目录是 Ceph，两者容量差两个数量级，装东西前要分清 |
| 5 | `nproc ; free -g` | 确认 128 核 / 755 G，决定 flash-attn 的 `MAX_JOBS` 和 GRPO 的 CPU offload 可行性 |

### 1.1 确认家目录位置

```bash
echo $HOME
ls -d ~/Temp-R1 2>/dev/null || echo "NO_REPO"
```
| # | 命令 | 作用 |
|---|---|---|
| 1 | `echo $HOME` | 手册全文用 `~` 指代运行根父目录，先确认真实路径 |
| 2 | `ls -d ~/Temp-R1` | 判断是否已拉过代码；输出 `NO_REPO` 说明要从第 3 节开始 |

### 本节验收

| 检查命令 | 期望输出 |
|---|---|
| `hostname` | `ecs-instance-cdb4917c` |
| `df -h / \| tail -1` | `overlay ... 457G` |
| `nproc` | `128` |

## 2. Miniconda 与 `tempr1` 环境

### 2.1 建环境（**两条命令，不要合并**）

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n tempr1 python=3.9 -y
conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit
conda list -n tempr1 | wc -l
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `source ~/miniconda3/etc/profile.d/conda.sh` | 未 init 的 conda 必须手动 source，否则后续 `conda`/`pip` 找不到 | 无报错 |
| 2 | `conda create -n tempr1 python=3.9 -y` | vllm 0.6.3 只保证 py3.8–3.10 可用，3.9 最稳 | `Executing transaction: done` |
| 3 | `conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit` | 提供 `nvcc`，flash-attn 若要编译时需要；与 torch cu121 版本对齐 | `nvcc --version` 出 12.1 |
| 4 | `conda list -n tempr1 \| wc -l` | 判断事务是否写完整；中断过的环境会只剩几十行或直接报错 | ≥ 90 |

### 2.2 torch（版本必须钉死 2.4.0）

```bash
conda activate tempr1
pip install -U pip setuptools wheel
pip install torch==2.4.0 --index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `conda activate tempr1` | 后续所有 pip 都装进这个环境 | 提示符带 `(tempr1)` |
| 2 | `pip install -U pip setuptools wheel` | 老 pip 解析 vllm 依赖会选错版本 | 无报错 |
| 3 | `pip install torch==2.4.0 --index-url .../tuna` | verl 的 FSDP 封装依赖 2.4 内部 API，2.5 会静默算错 | 下载约 2.5 GB |
| 4 | `python -c "import torch;..."` | 三个字段一次性验完 | `2.4.0+cu121 12.1 True` |

若镜像缺该版本，改用官方源：`pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121`。

### 2.3 vllm 0.6.3（会连带改 transformers，装完要回钉）

```bash
pip install vllm==0.6.3
python -c "import torch, vllm;print(torch.__version__, vllm.__version__)"
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `pip install vllm==0.6.3` | verl fork 的 rollout 只适配这一版，0.8.x 的构造参数与权重同步接口都变了 | 约 3 GB 装完 |
| 2 | `python -c "import torch, vllm;..."` | 确认 pip 没把 torch 顶到 2.5 | `2.4.0+cu121 0.6.3`（`vllm.__version__` 显示 `dev` 属正常，`pip show vllm` 仍是 0.6.3） |

### 2.4 flash-attn（用预编译轮子，不编译）

```bash
curl -L -o "/tmp/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl" "https://ghfast.top/https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"
pip install "/tmp/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl" --no-deps
python -c "import torch, flash_attn;print(flash_attn.__version__, torch._C._GLIBCXX_USE_CXX11_ABI)"
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `curl -L -o "/tmp/flash_attn-2.6.3+...whl" ...` | 直接下预编译轮子；文件名必须与官方一致，随便命名成 `fa.whl` 会被 pip 拒绝 | 约 179 MB |
| 2 | `pip install <whl> --no-deps` | `--no-deps` 防止它把 torch 重装一遍 | `Successfully installed` |
| 3 | `python -c "import torch, flash_attn;..."` | 校验 ABI 是否匹配（轮子名 `cxx11abiFALSE` 对应 `False`） | `2.6.3 False` |

若 `pip install flash-attn==2.6.3 --no-build-isolation` 报 `[Errno 18] Invalid cross-device link`，是本机 pip 缓存盘（Ceph）与临时目录跨设备，**不要重试编译**，直接走上面的下轮子方案。

### 2.5 LLaMA-Factory 与 verl 依赖（顺序敏感）

```bash
cd ~/Temp-R1
pip install -r requirements.txt "numpy<2"
cd ~/Temp-R1/LLaMA-Factory
pip install -e ".[metrics]"
pip install deepspeed==0.15.4
cd ~/Temp-R1
pip install "transformers==4.47.1" "tokenizers==0.21.1" --no-deps
pip check
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `pip install -r requirements.txt "numpy<2"` | verl 的运行时依赖（ray / tensordict / accelerate 等），文件里**不含 torch**，不会顶版本 | 无报错 |
| 2 | `pip install -e ".[metrics]"` | 装 LLaMA-Factory 本体；**禁止写 `.[torch]`**，那会重装 torch | 出现 `llamafactory 0.9.4.dev0` |
| 3 | `pip install deepspeed==0.15.4` | SFT 用 ZeRO-3 offload 配置，版本不对会读不懂 json | 无报错 |
| 4 | `pip install "transformers==4.47.1" "tokenizers==0.21.1" --no-deps` | 上一步把 transformers 升到 4.56.1，必须回钉；`--no-deps` 防止 resolver 再升回去 | 输出 `Successfully installed` |
| 5 | `pip check` | 元数据冲突（llamafactory 要求 ≥4.49、verl 要求 <4.48）无法消除，只能确认没有硬性缺依赖 | 不是 `No broken requirements` 也可接受，以 2.6 实跑为准 |

### 2.6 决定性验证：真跑一次 vllm

```bash
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-1.5B-Instruct', dtype='bfloat16', gpu_memory_utilization=0.5, enforce_eager=True)
out = llm.generate(['Hello, my name is'], SamplingParams(max_tokens=32, temperature=0))
print('OUTPUT:', out[0].outputs[0].text)
"
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | 上述 python 片段 | 元数据冲突只能靠实跑裁决；能加载并生成才说明 transformers/tokenizers/vllm 三方兼容 | 打印 `OUTPUT:` 后跟一段英文，加载显存约 2.9 GB |

### 2.7 记录环境快照

```bash
pip freeze > ~/Temp-R1/env_freeze_$(date +%Y%m%d).txt
conda list -n retriever > ~/Temp-R1/env_freeze_retriever_$(date +%Y%m%d).txt 2>/dev/null
```
| # | 命令 | 作用 |
|---|---|---|
| 1 | `pip freeze > env_freeze_<日期>.txt` | 复现出问题时靠它逐条回滚，比凭记忆强 |
| 2 | `conda list -n retriever > ...` | 检索环境单独存一份（第 5 节会建） |

### 本节验收

| 检查命令 | 期望输出 |
|---|---|
| `python -V` | `Python 3.9.25` |
| `nvcc --version \| tail -2` | `release 12.1, V12.1.66` |
| `python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` | `2.4.0+cu121 12.1 True` |
| `python -c "import vllm,flash_attn,deepspeed;print(flash_attn.__version__)"` | `2.6.3` |
| `python -c "import transformers;print(transformers.__version__)"` | `4.47.1` |

## 3. 拉取代码

```bash
cd ~
git clone --depth 1 https://ghfast.top/https://github.com/zjukg/Temp-R1.git Temp-R1
cd ~/Temp-R1 && du -sh . && ls
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `git clone --depth 1 https://ghfast.top/https://github.com/zjukg/Temp-R1.git Temp-R1` | 走 ghfast 镜像，直连 GitHub 在国内常断；`--depth 1` 省掉历史 | `580 files` |
| 2 | `du -sh . && ls` | 确认 corpus 随仓库落地（因此 978 M 远大于仓库标称 114 MB） | `978M`，含 `corpus/` |

### 3.1 仓库结构

| 目录/文件 | 用途 |
|---|---|
| `verl/` | 改造版训练框架（纯包，**无 setup.py**，靠 `PYTHONPATH=$HOME/Temp-R1` 使用） |
| `LLaMA-Factory/` | SFT 微调框架，唯一有 `setup.py` 的子目录 |
| `search_r1/` | E5 检索器、建索引、检索服务 |
| `scripts/data_process/` | 数据加工脚本（题型分类、过采样、轨迹生成、parquet 生成） |
| `scripts/multitq/` | GRPO stage1/stage2、评测脚本 |
| `corpus/` | `full_kg.jsonl` 461,329 条事实 + 索引产出 |
| `data/MultiTQ/` | 题目 json（仓库只带 dev/test） |

### 本节验收

| 检查命令 | 期望输出 |
|---|---|
| `ls ~/Temp-R1` | 含 `verl`、`LLaMA-Factory`、`search_r1`、`scripts`、`corpus` |
| `ls -lh ~/Temp-R1/data/MultiTQ/dev.json` | 20 M |
| `git -C ~/Temp-R1 log --oneline -1` | 一行 commit |

## 4. MultiTQ 数据

仓库只带 `dev.json` / `test.json`，**训练集 `train.json` 必须从 MultiTQ 官方仓库取**，这一步最容易漏。

### 4.1 下载与校验

```bash
curl -L -o /tmp/multitq.zip "https://ghfast.top/https://github.com/czy1999/MultiTQ/raw/main/data/Dataset.zip"
mkdir -p /tmp/multitq_peek
unzip -o /tmp/multitq.zip -d /tmp/multitq_peek > /dev/null
find /tmp/multitq_peek -name "*.json" -path "*questions*" -exec ls -lh {} \;
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `curl -L -o /tmp/multitq.zip ...` | 从 ACL 2023 MultiTQ 官方仓库取 `data/Dataset.zip` | 约 39 MB |
| 2 | `mkdir -p /tmp/multitq_peek` | 先解到 `/tmp` 再看，避免直接覆盖 `data/MultiTQ/` 里已有的 dev/test | 目录建立 |
| 3 | `unzip -o ... -d /tmp/multitq_peek` | 展开题目与 KG 文件 | 无 `error` |
| 4 | `find ... -name "*.json" -path "*questions*"` | 确认三个题目文件都在 | `train.json` 131 M、`dev.json` 20 M、`test.json` 19 M |

### 4.2 拷入并对齐版本

```bash
cp /tmp/multitq_peek/MultiTQ/questions/{train,dev,test}.json ~/Temp-R1/data/MultiTQ/
md5sum /tmp/multitq_peek/MultiTQ/questions/dev.json ~/Temp-R1/data/MultiTQ/dev.json
md5sum /tmp/multitq_peek/MultiTQ/questions/test.json ~/Temp-R1/data/MultiTQ/test.json
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `cp .../{train,dev,test}.json ~/Temp-R1/data/MultiTQ/` | 三件套落到运行根；dev/test 覆盖为官方同版本 | 三个文件存在 |
| 2 | `md5sum` 两行对两次 | 防止"看着像同一份、实际差几个样本"导致标签错位 | 每对 md5 相同（dev `ec2bfbd5…`、test `cb24b44a…`） |

### 4.3 按题型切分

```bash
cd ~/Temp-R1
~/miniconda3/envs/tempr1/bin/python scripts/data_process/qtype_classify.py --input data/MultiTQ/train.json --output_dir data/MultiTQ/split_by_qtype
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `qtype_classify.py --input ... --output_dir ...` | 按 `qtype` 分成 6 组，供过采样按题型配比取样 | 打印 `Split 386787 examples into 6 qtype groups` |

### 4.4 过采样出 SFT 候选

```bash
cd ~/Temp-R1
~/miniconda3/envs/tempr1/bin/python scripts/data_process/create_oversampled_dataset.py
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `create_oversampled_dataset.py` | 按各题型"预期准确率"倒推样本量（准确率低的题型多采），`seed=42` 可复现 | 输出 `Total samples: 1,885` |

### 4.5 生成 GRPO / 评测用的 parquet

脚本第 307 行写死读 `train_MULTITQ.json`，而官方文件名是 `train.json`。**复制一份改名，不建软链接**：

```bash
cd ~/Temp-R1
cp data/MultiTQ/train.json data/MultiTQ/train_MULTITQ.json
PYTHONPATH=$HOME/Temp-R1 ~/miniconda3/envs/tempr1/bin/python scripts/data_process/multitq_search.py --input_dir data/MultiTQ --local_dir data/multitq_search2 --template_type base
ls -lh data/multitq_search2
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `cp data/MultiTQ/train.json data/MultiTQ/train_MULTITQ.json` | 满足脚本的硬编码文件名；复制而非链接，删起来互不影响 | 131 M 副本 |
| 2 | `PYTHONPATH=$HOME/Temp-R1 python multitq_search.py --template_type base` | `verl` 是无 setup 的纯包，必须给 `PYTHONPATH`；`base` 模板含 `<plan>/<filter>/<rank>` 动作定义，**不能改** | 打印 `Processing complete!` |
| 3 | `ls -lh data/multitq_search2` | 确认 6 个 parquet 齐全 | 合计 499 M |

### 4.6 数据基线（2026-11-08 实测，用于校验你的拷贝）

| 子集 | 条数 | 占比 |
|---|---|---|
| train_all | 386,787 | 100% |
| train_single | 283,482 | 73.3% |
| train_multiple | 103,305 | 26.7% |
| test_all | 54,584 | — |
| test_single | 38,864 | 71.2% |
| test_multiple | 15,720 | 28.8% |

题型分布：`equal` 135,890、`before_after` 75,340、`first_last` 72,252、`after_first` 43,305、`before_last` 43,107、`equal_multi` 16,893。
`single + multiple = 386,787` 说明 `qlabel` 只有两种取值、没有样本被静默丢弃；若你的结果小于全量，说明数据版本不一致。

### 本节验收

| 检查命令 | 期望输出 |
|---|---|
| `ls -lh ~/Temp-R1/data/MultiTQ` | 含 `train.json` 131 M、`train_MULTITQ.json` 131 M |
| `ls ~/Temp-R1/data/MultiTQ/split_by_qtype` | 6 个 `train_<qtype>.json` + `qtype_statistics.json` |
| `ls -lh ~/Temp-R1/data/multitq_search2` | 6 个 parquet，合计 499 M |

## 5. 检索链路

### 5.1 建 `retriever` 环境

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n retriever python=3.10 -y
conda activate retriever
pip install torch==2.4.0 --index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install faiss-gpu-cu12==1.8.0
pip install "numpy==1.26.4"
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `conda create -n retriever python=3.10 -y` | 检索与训练依赖冲突（faiss 要求 numpy<2 等），必须分环境 | 建好 |
| 2 | `pip install torch==2.4.0 ...` | E5 编码需要 torch；与主环境同版本便于排错 | `2.4.0+cu121` |
| 3 | `pip install faiss-gpu-cu12==1.8.0` | **包名不是 `faiss-gpu`**，那个名字止于 1.7.2、装 1.8.0 会 `No matching distribution` | `import faiss` 出 `1.8.0` |
| 4 | `pip install "numpy==1.26.4"` | faiss 拉进来的 numpy 2.x 会触发 `_ARRAY_API not found`，必须回钉 | 无报错 |

### 5.2 其余检索依赖

```bash
conda activate retriever
pip install "transformers==4.47.1" "tokenizers==0.21.1" "datasets<3" tqdm fastapi uvicorn "pydantic<3"
python -c "import faiss, torch, transformers;print(faiss.__version__, torch.__version__, transformers.__version__)"
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `pip install ...` | 只装脚本真正 import 的包；**不需要** sentence-transformers，也**不需要** pyserini/Java | 无报错 |
| 2 | `python -c "import faiss, ..."` | 确认 faiss 能调 GPU | `1.8.0 2.4.0+cu121 4.47.1` |

### 5.3 建索引

脚本原为绝对路径（`cd /search_r1/search`、`/corpus/...`），本机已重写为 `~/Temp-R1` 下相对路径。换机器时按下面这份覆盖：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate retriever
export HF_ENDPOINT=https://hf-mirror.com
cd ~/Temp-R1/search_r1/search
~/miniconda3/envs/retriever/bin/python index_builder.py --retrieval_method e5 --model_path intfloat/e5-base-v2 --corpus_file ~/Temp-R1/corpus/full_kg.jsonl --save_dir ~/Temp-R1/corpus/kg_index --use_fp16 --max_length 256 --batch_size 512 --pooling_method mean --faiss_type Flat --save_embedding
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `conda activate retriever` + `export HF_ENDPOINT=https://hf-mirror.com` | E5 权重从 HF 拉，国内直连常超时 | 无 401/超时 |
| 2 | `index_builder.py ...` | 46 万条事实用 e5-base-v2 编码后建 Flat 索引；`--save_embedding` 留缓存便于重建 | `e5_Flat.index` 1.4 G + `emb_e5.memmap` 1.4 G |
| 3 | 参数 `--faiss_type Flat --max_length 256 --pooling_method mean` | 与论文检索器口径一致，改了会导致召回分布偏移 | 见 5.4 验收 |

**参数名以 `--help` 为准**：若 `index_builder.py --help` 里没有 `--retrieval_method`，删掉该参数即可，其余保持。

### 5.4 起检索服务（必须常驻）

```bash
cd ~/Temp-R1
nohup bash retrieval_launch.sh > ~/Temp-R1/retrieval.log 2>&1 &
sleep 20
tail -5 ~/Temp-R1/retrieval.log
curl -s -X POST http://127.0.0.1:8000/retrieve -H "Content-Type: application/json" -d '{"queries":["Who was the last person to visit China before the University of Stellenbosch?"],"topk":3,"return_scores":true}' | head -c 400
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `nohup bash retrieval_launch.sh &` | 服务常驻；SFT 轨迹生成与 GRPO 每次 `<search>` 都依赖它，断了训练就卡 | 进程在 |
| 2 | `tail -5 retrieval.log` | 看是否成功加载 1.4 G 索引 | 含 `Uvicorn running on http://0.0.0.0:8000` |
| 3 | `curl ... /retrieve` | 用论文 Figure 2 的原题探活，验证索引与语料对齐 | 返回 `Stellenbosch_University` 相关三元组 |

### 本节验收

| 检查命令 | 期望输出 |
|---|---|
| `ls -lh ~/Temp-R1/corpus/kg_index` | `e5_Flat.index` 1.4 G、`emb_e5.memmap` 1.4 G |
| `pgrep -af retrieval_server` | 一行 python 进程 |
| `curl -s -X POST .../retrieve ...` | JSON 含 `Stellenbosch_University`，score 约 0.85 |

## 6. 脚本路径与并行适配

上游脚本按"4 卡 + anaconda3 + 绝对路径"写的，本机是"单卡 + miniconda3 + `~/Temp-R1`"。**只改这些，不动超参。**

### 6.1 三处 sed（改完立刻可执行）

```bash
sed -i 's#~/anaconda3#~/miniconda3#g' ~/Temp-R1/scripts/multitq/train_grpo_stage1.sh ~/Temp-R1/scripts/multitq/train_grpo_stage2.sh ~/Temp-R1/scripts/multitq/evaluate.sh
grep -n "conda.sh" ~/Temp-R1/scripts/multitq/*.sh
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `sed -i 's#~/anaconda3#~/miniconda3#g' ...` | 三个脚本首行 `source ~/anaconda3/...` 在本机不存在，不改连第一行都过不去 | 无报错 |
| 2 | `grep -n "conda.sh" ...` | 确认全变成 `~/miniconda3` | 三行全是 miniconda3 |

### 6.2 单卡适配清单（逐项核对，不认识的项先别改）

| 文件 | 变量 | 上游值 | 本机值 | 说明 |
|---|---|---|---|---|
| `train_grpo_stage1.sh` | `CUDA_VISIBLE_DEVICES` | `1,2,3` | `0` | 本机只有 cuda:0，写 1,2,3 会找不到设备 |
| `train_grpo_stage1.sh` | `trainer.n_gpus_per_node` | `3` | `1` | 与上一条必须同时改 |
| `train_grpo_stage1.sh` | `BASE_MODEL` | 8B SFT 产物 | `Qwen/Qwen2.5-1.5B-Instruct` 或 SFT 输出目录 | **需在脚本 40 行后确认实际写法** |
| `train_grpo_stage2.sh` | `STAGE1_CHECKPOINT` | `verl_checkpoints/xxx` | stage1 实际目录 | 占位符，stage1 跑完才知道 |
| `train_grpo_stage2.sh` | `CUDA_VISIBLE_DEVICES` / `n_gpus_per_node` | `3,4,5` / `2` | `0` / `1` | 同上 |
| `evaluate.sh` | `CUDA_VISIBLE_DEVICES` | `1,2,3` | `0` | 评测单卡即可 |

**未核对项**（写作时尚未看到脚本 40 行之后的内容，执行前必须 `sed -n '40,110p'` 自查）：`BASE_MODEL` 的确切写法、`n_gpus_per_node` 的参数名、`rollout.n` 是否与 `n_agent` 绑定。

### 6.3 绝对不能改的（改了就不是复现）

| 项 | 位置 | 为什么不能动 |
|---|---|---|
| `--template_type base` | 4.5 节 | 提示词里 `<plan>/<filter>/<rank>` 的动作定义来自这里，换了模型不会用 `<filter>` |
| `kl_loss_coef` | stage1 `0.01` / stage2 `0.001` | 论文正文写 0.01、代码 stage2 写 0.001，**以代码为准并在报告里记录**，不要"统一" |
| `n_agent` | stage1 `5` / stage2 `8` | 与论文的 G=5 不一致，但这是官方可跑配置，改回 5 会让 stage2 显存与步数全变 |
| FSDP `param/grad/optimizer_offload` | 两个 GRPO 脚本 | 正是靠全 offload，单卡 48 G 才能跑；关掉直接 OOM |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 各脚本 | 长 context 多轮 rollout 必须靠它压碎片，删了会在第 2–3 轮 OOM |

### 本节验收

| 检查命令 | 期望输出 |
|---|---|
| `grep -c "miniconda3" ~/Temp-R1/scripts/multitq/train_grpo_stage1.sh` | ≥ 1 |
| `grep -n "CUDA_VISIBLE_DEVICES" ~/Temp-R1/scripts/multitq/*.sh` | 三个文件均为 `0` |
| `bash -n ~/Temp-R1/scripts/multitq/train_grpo_stage1.sh` | 无输出（语法通过） |

## 7. 生成 1,000 条 SFT 冷启动轨迹

这一步是**唯一的外部依赖**：需要一个 OpenAI 兼容的强模型（论文用 GPT-4o，一次性约 $50）。

### 7.1 传参方式

`generate_sft_trajectories.py` 支持两种后端：

| 后端 | 参数 | 说明 |
|---|---|---|
| 云端 | `--model_api openai --api_key <key> --base_url <url> --model <name>` | 任何 OpenAI 兼容接口（含 DeepSeek、Qwen、Kimi）都能用 |
| 本地 | `--model_api local --model_url http://localhost:8001/v1/chat/completions --model <name>` | 自己起 vLLM 当 teacher，质量会掉 |

### 7.2 跑

```bash
cd ~/Temp-R1
export OPENAI_API_KEY=<your-key>
export OPENAI_BASE_URL=<https://api.openai.com/v1 或兼容地址>
~/miniconda3/envs/tempr1/bin/python scripts/data_process/generate_sft_trajectories.py --input data/MultiTQ/oversampled_for_sft.json --output outputs/train_sft_oversampled_merged.jsonl --num_samples 1000 --start_index 0 --retriever_url http://127.0.0.1:8000/retrieve --model_api openai --api_key "$OPENAI_API_KEY" --base_url "$OPENAI_BASE_URL" --model <model-name> --filter_correct_only
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `export OPENAI_API_KEY=...` | 客户端要求非空；兼容接口不看内容但要求有 | 无报错 |
| 2 | `--retriever_url http://127.0.0.1:8000/retrieve` | 轨迹里的每次检索都打到本机服务上（第 5 节那个），所以它必须先起着 | 服务在监听 |
| 3 | `--num_samples 1000 --filter_correct_only` | 只保留"答对"的轨迹，1,885 条候选按命中率截到 1,000 | 输出行数 ≈ 1000 |
| 4 | `--model <name>` | 填具体模型名；留空会用默认 `gpt-4o` | 与 key 匹配 |

### 7.3 转成 LLaMA-Factory 格式

```bash
cd ~/Temp-R1
~/miniconda3/envs/tempr1/bin/python scripts/data_process/llama-factory-dataset.py
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `llama-factory-dataset.py` | 把轨迹 jsonl 转成 `dataset_info.json` + alpaca 风格数据，供 SFT yaml 的 `dataset: multitq_sft_train` 引用 | 生成数据文件，条数同 7.2 |

**未核对**：该脚本的参数（是否需 `--input/--output`）尚未确认，执行前先 `sed -n '1,80p'` 或 `--help`。

### 本节验收

| 检查命令 | 期望输出 |
|---|---|
| `wc -l ~/Temp-R1/outputs/train_sft_oversampled_merged.jsonl` | 约 1000 |
| `head -c 500 ~/Temp-R1/outputs/train_sft_oversampled_merged.jsonl` | 含 `<plan>` / `<search>` / `<answer>` 标记 |

## 8. SFT 训练（1.5B）

### 8.1 只改 yaml 里的两行

`LLaMA-Factory/examples/train_full/tempr1_full_sft.yaml` 保持其余不动（`finetuning_type: full`、`cutoff_len: 8192`、`max_samples: 1000`、`per_device_train_batch_size: 5`、`gradient_accumulation_steps: 4`、`learning_rate: 2.0e-5`、`num_train_epochs: 2.0`、`lr_scheduler_type: cosine`、`warmup_ratio: 0.1`、`bf16: true`、`gradient_checkpointing: true`、`flash_attn: fa2`、`deepspeed: ds_z3_offload_config.json`）：

```yaml
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
template: qwen
```
| # | 改动 | 原因 |
|---|---|---|
| 1 | `model_name_or_path` 换成 1.5B | 单卡 48 G 唯一能零妥协跑全参的档 |
| 2 | `template: llama` → `qwen` | Qwen2.5 的对话模板与 LLaMA 不同，写错 SFT 学不到东西 |

### 8.2 启动

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tempr1
cd ~/Temp-R1/LLaMA-Factory
export CUDA_VISIBLE_DEVICES=0
llamafactory-cli train examples/train_full/tempr1_full_sft.yaml
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `conda activate tempr1` | SFT 在 `tempr1` 环境里，不是 `retriever` | 提示符 `(tempr1)` |
| 2 | `export CUDA_VISIBLE_DEVICES=0` | 单卡 | 无报错 |
| 3 | `llamafactory-cli train ...` | 全参 + ZeRO-3 offload；1,000 条 × 2 epoch | 末尾出现 loss 且保存 checkpoint |

显存参考：1.5B 全参 AdamW 状态约 24.6 GB，加 π_ref 与激活，峰值 41–45 GB，48 G 卡可跑。

### 本节验收

| 检查命令 | 期望输出 |
|---|---|
| `ls ~/Temp-R1/LLaMA-Factory/saves/` | 含新增实验目录与 `checkpoint-*` |
| `grep -c "loss" <训练日志>` | 逐步下降而非 NaN |

## 9. GRPO 两阶段

### 9.1 阶段一：只在 multiple 题上训（逆课程第一段）

```bash
cd ~/Temp-R1
bash scripts/multitq/train_grpo_stage1.sh
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `bash scripts/multitq/train_grpo_stage1.sh` | 读 `train_multiple.parquet`（103,305 条），80 步 × batch 60 × G=5 | 日志出现 step 递增、无 NCCL 报错 |

关键参数（来自代码，与本机一致）：`train_batch_size=60`、`ppo_mini_batch_size=30`、`ppo_micro_batch_size=3`、`lr=5e-7`、`kl_loss_coef=0.01`、`n_agent=5`、`temperature=1.0`、`gpu_memory_utilization=0.65`、`total_training_steps=80`、`max_turns=3`、`retriever.topk=30`、`max_response_length=1000`。

### 9.2 阶段二：全量续训

```bash
cd ~/Temp-R1
ls verl_checkpoints/
sed -i "s#STAGE1_CHECKPOINT='verl_checkpoints/xxx'#STAGE1_CHECKPOINT='<上一步实际目录>'#" scripts/multitq/train_grpo_stage2.sh
bash scripts/multitq/train_grpo_stage2.sh
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `ls verl_checkpoints/` | 拿到 stage1 实际产出目录名 | 含 `global_step_80` |
| 2 | `sed -i "s#...#..."` | 把占位符替换成真路径 | `grep STAGE1_CHECKPOINT` 显示真路径 |
| 3 | `bash .../train_grpo_stage2.sh` | 从 stage1 续训全量，500 步、`resume_step=40`、`n_agent=8` | 日志 step 从 40 起递增 |

### 本节验收

| 检查命令 | 期望输出 |
|---|---|
| `nvidia-smi --query-gpu=memory.used --format=csv` | 训练中 30–45 GB（全 offload 后 GPU 侧不高） |
| `grep -c "global_step" <日志> \| tail -1` | 递增到 500 |

## 10. 评测

### 10.1 三档对照

```bash
cd ~/Temp-R1
bash scripts/multitq/evaluate.sh
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `bash scripts/multitq/evaluate.sh` | `+trainer.val_only=true`，500 题、`n_agent=1`、`topk=30` | 打印 accuracy |

评测口径：`val_batch_size=500`、`val_data_num=500`、`max_prompt_length=6200`、`max_response_length=5000`、`max_obs_length=2400`。

**注意**：脚本开头会 `pkill -9 -u $USER ray` / `vllm`，跑之前确认没有别的训练在跑。

### 10.2 复现锚点（论文 Figure 4 / 5，PDF 第 7 页）

| 档位 | 论文值 | 允许偏差 |
|---|---|---|
| Base（Qwen2.5-1.5B-Instruct） | 0.210 | ±0.02 |
| + SFT | 0.374 | ±0.02 |
| + GRPO | 0.532 | ±0.03 |

三档是**阶梯**关系：只要求"递增"和"增量接近"（+0.164 / +0.158），不要求单点精确——论文没给 seed、没给 std，单点精度本来就不可复现。

### 本节验收

| 检查命令 | 期望输出 |
|---|---|
| `grep -i "accuracy" <评测日志>` | 三档依次约 0.21 / 0.374 / 0.532 |

## 11. 常见错误对照

| # | 报错原文 | 真因 | 处置 |
|---|---|---|---|
| 1 | `bash: pip: command not found` | 新 shell 没 source conda | `source ~/miniconda3/etc/profile.d/conda.sh && conda activate tempr1` |
| 2 | `conda: error: unrecognized arguments: cuda-toolkit` | 用 `\` 续行把两条命令拼成一条 | 拆成两行分别执行 |
| 3 | `DirectoryNotACondaEnvironmentError: The target directory exists, but it is not a conda environment` | 中断的 `conda create` 留下写了一半的 `conda-meta` | `rm -rf ~/miniconda3/envs/<name>` 后重建 |
| 4 | `error: [Errno 18] Invalid cross-device link` | pip 缓存盘（Ceph）与 `/tmp` 跨设备 rename | 不编译，直接下预编译 whl（2.4 节） |
| 5 | `ERROR: Invalid wheel filename (wrong number of parts)` | 把 whl 存成了 `fa.whl` | 用官方原文件名 |
| 6 | `ERROR: No matching distribution found for faiss-gpu==1.8.0` | 包名不存在，`faiss-gpu` 止于 1.7.2 | 用 `faiss-gpu-cu12==1.8.0` |
| 7 | `AttributeError: _ARRAY_API not found` / `numpy.core.multiarray failed to import` | numpy 2.x 与旧 faiss 不兼容 | `pip install "numpy==1.26.4"` |
| 8 | `ImportError: tokenizers>=0.21,<0.22 is required` | 装 LLaMA-Factory 时 tokenizers 被升到 0.22 | `pip install "transformers==4.47.1" "tokenizers==0.21.1" --no-deps` |
| 9 | `FileNotFoundError: data/MultiTQ/train_MULTITQ.json` | `multitq_search.py` 第 307 行硬编码文件名 | 复制 `train.json` 为 `train_MULTITQ.json`（4.5 节） |
| 10 | `No such file or directory: ~/anaconda3/etc/profile.d/conda.sh` | 三个训练脚本写死 anaconda3 | 6.1 节 sed 改 miniconda3 |
| 11 | `RuntimeError: Found no NVIDIA driver` / `CUDA error: invalid device ordinal` | `CUDA_VISIBLE_DEVICES=1,2,3` 但机器只有一张卡 | 改 `0`，同时 `n_gpus_per_node=1` |
| 12 | `torch.OutOfMemoryError` 于第 2–3 轮 rollout | 长 context 碎片或关掉了 offload | 保留 `expandable_segments:True`，别关 FSDP offload |
| 13 | conda 卡在 `Executing transaction`、进程 `Dl+` | Ceph 元数据压力导致假死，不是命令错 | `timeout 5 ls ~` 探活；必要时杀掉重建 |
| 14 | `bash: conda: command not found` 在 nohup 脚本里 | 脚本内没 source conda | 用绝对解释器 `~/miniconda3/envs/<env>/bin/python` |

## 12. tmux 速查与换机器

### 12.1 长任务必须放 tmux

```bash
tmux new -d -s stage1 -c ~/Temp-R1 'bash scripts/multitq/train_grpo_stage1.sh 2>&1 | tee logs/stage1.log'
sleep 20
tmux ls ; pgrep -af main_ppo | head -3 ; tail -20 ~/Temp-R1/logs/stage1.log
```
| # | 命令 | 作用 | 成功判据 |
|---|---|---|---|
| 1 | `tmux new -d -s stage1 ... 'bash ... \| tee logs/stage1.log'` | 断连不断训；日志同时落盘 | 返回 `[detached]` |
| 2 | `sleep 20` + `tmux ls` | 体检：只凭 `[1] <pid>` 判断成功会被假启动骗 | session 存在 |
| 3 | `pgrep -af main_ppo` | 确认训练进程真的起来了 | 一行 python |
| 4 | `tail -20 logs/stage1.log` | 确认进入 step 循环而非停在导入 | 有 step 行 |

### 12.2 tmux 速查

| 场景 | 命令 |
|---|---|
| 建后台会话 | `tmux new -d -s <name> -c ~/Temp-R1 '<cmd>'` |
| 列会话 | `tmux ls` |
| 附着 / 脱离 | `tmux attach -t <name>` / `Ctrl-b d` |
| 看最后输出 | `tmux capture-pane -p -t <name> \| tail -30` |
| 杀掉 | `tmux kill-session -t <name>` |
| 清理 ray 残留 | `pkill -9 -u $USER ray; rm -rf /tmp/ray_$USER* ~/.ray` |

### 12.3 换机器清单

| # | 项 | 做法 |
|---|---|---|
| 1 | 确认卡空闲 | `nvidia-smi`，有他人进程就换机（`e1330fb1` 长期被占） |
| 2 | conda 路径 | 若为 `~/anaconda3`，跳过 6.1 的 sed |
| 3 | 显卡数 | 单卡改 `CUDA_VISIBLE_DEVICES=0` + `n_gpus_per_node=1`；多卡按 0,1,2… |
| 4 | 内存门槛 | 3B 全 offload 需 ≥64 G 主机内存；1.5B 需 ≥32 G |
| 5 | 重建索引 | Ceph 跨机可共享 `corpus/`，但 faiss 版本必须一致，否则重跑 5.3 |
| 6 | 数据 | `data/MultiTQ/` 与 `data/multitq_search2/` 可直接拷，不必重跑第 4 节 |

### 本节验收

| 检查命令 | 期望输出 |
|---|---|
| `tmux ls` | 列出正在跑的会话 |
| `pgrep -af retrieval_server` | 检索服务仍在（换机器后需重启） |

## 附：本手册的事实来源

| 结论 | 依据 |
|---|---|
| 环境版本号 | 2026-11-08 会话中 `python -V`、`nvcc --version`、`conda list \| wc -l`、`pip show` 实测输出 |
| 数据条数 386,787 / 57,979 / 54,584 | `qtype_classify.py` 与 `multitq_search.py` 的实际打印 |
| parquet 六个子集条数 | `multitq_search.py` 末尾打印的 `Dataset Statistics` |
| 索引 461,329 条 / dim 768 | `corpus/full_kg.jsonl` 行数与 `index_builder.py` 打印 |
| 检索验收 | `curl POST /retrieve` 返回 `Stellenbosch_University ... 0.8566` |
| 论文超参与锚点 | `paper_text.txt` 第 7 页（Figure 4 读数）与附录 C（3×A800、AdamW、FSDP、梯度裁剪 5.0） |
| G / β / batch 的代码与论文冲突 | `train_grpo_stage1.sh` / `stage2.sh` 与论文 §4 的逐项比对 |

**未核对项汇总**（执行前必须自查，不要当成已确认）：

1. `train_grpo_stage1.sh` 第 40 行之后的 `BASE_MODEL` 写法与 `n_gpus_per_node` 参数名。
2. `llama-factory-dataset.py` 的入参形式。
3. `generate_sft_tra.sh` 与 `generate_sft_trajectories.py` 两套入口以哪套为准（后者参数更全，建议用后者）。
4. `multitq_search.py` 中 `--hdfs_dir` 是否需要显式置空。
