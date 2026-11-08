#!/usr/bin/env bash
set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tempr1
cd "$HOME/Temp-R1/LLaMA-Factory"

export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export FORCE_TORCHRUN=1
export NPROC_PER_NODE=1
export OMP_NUM_THREADS=4
unset DISABLE_VERSION_CHECK

mkdir -p ../logs
python -m llamafactory.cli train examples/train_full/tempr1_qwen_1p5b_sft.yaml 2>&1 | tee ../logs/sft_qwen_1p5b.log
