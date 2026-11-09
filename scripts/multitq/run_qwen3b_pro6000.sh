#!/usr/bin/env bash
# Upload to ~/Temp-R1/scripts/multitq/run_qwen3b_pro6000.sh.
# Upload the YAML to ~/Temp-R1/LLaMA-Factory/examples/train_full/.
# From ~: bash ~/Temp-R1/scripts/multitq/run_qwen3b_pro6000.sh prepare
# On the RTX PRO 6000: bash ~/Temp-R1/scripts/multitq/run_qwen3b_pro6000.sh train
set -eo pipefail

ROOT="$HOME/Temp-R1"
FACTORY="$ROOT/LLaMA-Factory"
CONFIG="$FACTORY/examples/train_full/tempr1_qwen_3b_pro6000_sft.yaml"
CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
ENV_DIR="$HOME/miniconda3/envs/tempr1-3b-pro6000"
PY="$ENV_DIR/bin/python"
SESSION=sft-qwen3b-pro6000

fail() { printf '停止：%s\n' "$*" >&2; exit 1; }

[[ -s "$CONFIG" ]] || fail "未找到 3B 配置：$CONFIG"
[[ -s "$FACTORY/data/multitq_sft_train.json" ]] || fail "未找到现有 MultiTQ SFT 数据。"
[[ -s "$FACTORY/src/llamafactory/launcher.py" ]] || fail "未找到项目中的 LLaMA-Factory 启动器。"
[[ -s "$CONDA_SH" ]] || fail "未找到 Miniconda：$CONDA_SH"
export PYTHONNOUSERSITE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
unset PYTHONPATH DISABLE_VERSION_CHECK

lock_environment() {
    mkdir -p "$(dirname -- "$ENV_DIR")"
    command -v flock >/dev/null || fail "找不到 flock，无法保护共享的训练环境。"
    exec 9>"$ENV_DIR.lock"
    flock -n 9 || fail "3B 环境正在准备或训练，请查看已有任务。"
}

check_assets() {
    "$PY" - "$CONFIG" "$FACTORY" "$1" <<'PY'
import json
from pathlib import Path
import shutil
import sys

from huggingface_hub import snapshot_download
import torch
from yaml import safe_load

config_path, factory, mode = sys.argv[1:]
factory = Path(factory)
cfg = safe_load(Path(config_path).read_text(encoding="utf-8"))
if torch.__version__.split("+")[0] != "2.7.1" or torch.version.cuda != "12.8":
    raise SystemExit("需要独立 3B 环境中的 PyTorch 2.7.1 + CUDA 12.8。")

# Import the actual LLaMA-Factory training entry before loading the model.
from llamafactory.train.tuner import run_exp

dataset_info = json.loads((factory / "data/dataset_info.json").read_text(encoding="utf-8"))
dataset = dataset_info.get(cfg["dataset"], {})
if dataset.get("file_name") != "multitq_sft_train.json" or dataset.get("formatting") != "sharegpt":
    raise SystemExit("MultiTQ 数据注册与现有 SFT 配方不一致。")

snapshot = Path(snapshot_download(
    repo_id=cfg["model_name_or_path"],
    revision=cfg["model_revision"],
    allow_patterns=["*.json", "*.safetensors", "merges.txt"],
    local_files_only=mode != "prepare",
))
model = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
if (model.get("model_type"), model.get("hidden_size"), model.get("num_hidden_layers")) != ("qwen2", 2048, 36):
    raise SystemExit("缓存的模型结构与 Qwen2.5-3B 不一致。")
index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
shards = set(index["weight_map"].values())
if not shards or any(not (snapshot / shard).is_file() or (snapshot / shard).stat().st_size == 0 for shard in shards):
    raise SystemExit("3B 权重分片不完整，请重新执行 prepare。")
if not (snapshot / "tokenizer.json").is_file():
    raise SystemExit("3B tokenizer 缓存不完整，请重新执行 prepare。")
print(f"3B 权重缓存可用：{snapshot}", flush=True)

if mode == "prepare":
    raise SystemExit(0)

if not torch.cuda.is_available():
    raise SystemExit("PyTorch 无法使用 CUDA，请检查当前节点的 NVIDIA 驱动。")
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
free, total = torch.cuda.mem_get_info(0)
if "RTX PRO 6000" not in name or capability != (12, 0) or total < 90000 * 1024**2:
    raise SystemExit(f"需要完整 RTX PRO 6000 Blackwell 96GB，当前设备：{name}，CC {capability}，{total / 1024**3:.1f} GiB。")
if free < 80 * 1024**3:
    raise SystemExit(f"空闲显存 {free / 1024**3:.1f} GiB，3B 全量 SFT 启动门槛为 80 GiB。")
if shutil.disk_usage(factory).free < 80 * 1024**3:
    raise SystemExit("项目所在磁盘至少需要 80 GiB 空闲空间保存 3B 训练状态。")

q, k, v = [torch.randn((1, 4, 512, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
result = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
result.float().square().mean().backward()
torch.cuda.synchronize()
if not all(t.grad is not None and torch.isfinite(t.grad).all().item() for t in (q, k, v)):
    raise SystemExit("BF16 SDPA 的反向计算结果无效。")
print(f"PRO6000_SFT_PREFLIGHT_OK {name} | {free / 1024**3:.1f} GiB 可用", flush=True)
PY
}

case "${1:-}" in
    prepare)
        lock_environment
        source "$CONDA_SH"
        if [[ ! -x "$PY" ]]; then
            conda create -y --prefix "$ENV_DIR" python=3.10 pip
        fi
        conda activate "$ENV_DIR"
        "$PY" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1
        "$PY" -m pip install "$FACTORY" \
            transformers==4.49.0 datasets==4.0.0 accelerate==1.10.1 \
            peft==0.17.1 trl==0.9.6 tensorboard==2.20.0
        "$PY" -m pip check
        check_assets prepare
        printf '3B 环境和权重已准备。\n'
        ;;
    train)
        [[ -x "$PY" ]] || fail "请先执行 prepare。"
        command -v tmux >/dev/null || fail "找不到 tmux。"
        if tmux has-session -t "=$SESSION" 2>/dev/null; then
            fail "当前节点的 $SESSION 已在运行。"
        fi
        NODE="$(hostname)"
        NODE="${NODE//[^a-zA-Z0-9._-]/_}"
        mkdir -p "$ROOT/logs"
        LOG="$ROOT/logs/sft_qwen3b_${NODE}_$(date +%Y%m%d-%H%M%S)_$$.log"
        SELF="$(realpath -- "${BASH_SOURCE[0]}")"
        tmux new-session -d -s "$SESSION" -c "$ROOT" \
            /usr/bin/env PATH="$PATH" HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HUB_CACHE" HF_ENDPOINT="$HF_ENDPOINT" \
            /bin/bash "$SELF" _worker "$LOG"
        printf '已提交到 tmux：%s\n日志：%s\n' "$SESSION" "$LOG"
        ;;
    _worker)
        [[ -n "${2:-}" ]] || fail "缺少本次训练日志路径。"
        (
            lock_environment
            source "$CONDA_SH"
            conda activate "$ENV_DIR"
            cd -- "$FACTORY"
            unset FORCE_TORCHRUN NPROC_PER_NODE
            export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
            check_assets train
            # This LLaMA-Factory version requires the distributed launcher even on one GPU.
            "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 \
                "$FACTORY/src/llamafactory/launcher.py" "$CONFIG"
        ) 2>&1 | tee -a -- "$2"
        ;;
    *)
        printf '用法：bash %s {prepare|train}\n' "$0" >&2
        exit 2
        ;;
esac
