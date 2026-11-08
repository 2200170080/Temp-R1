#!/usr/bin/env bash
# Single-A6000 GRPO Stage1: 80 updates from the full SFT model.
set -eo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tempr1
cd "$HOME/Temp-R1"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

BASE_MODEL="$PWD/LLaMA-Factory/saves/qwen2.5-1.5b-it/full/sft"
EXPERIMENT_NAME="qwen1p5b-stage1-$(date +%Y%m%d-%H%M%S)"
test -s "$BASE_MODEL/model.safetensors"
test -s data/multitq_search2/train_multiple.parquet
mkdir -p logs

# The original trainer increments the counter before checking >=, so a limit
# of N performs N-1 updates. Fix the comparison to perform exactly N updates.
# No optimizer, reward, or model configuration is changed by this patch.
python - <<'PY'
from pathlib import Path
path = Path("verl/trainer/ppo/ray_trainer.py")
text = path.read_text(encoding="utf-8")
old = "if self.global_steps >= self.total_training_steps:"
new = "if self.global_steps > self.total_training_steps:"
if text.count(old) == 1 and new not in text:
    updated = text.replace(old, new, 1)
    compile(updated, str(path), "exec")
    path.write_text(updated, encoding="utf-8")
    print("Fixed trainer stop condition: configured N now executes N updates.")
elif text.count(new) == 1 and old not in text:
    print("Trainer stop condition already fixed.")
else:
    raise SystemExit("Trainer differs from expected source; stopped without patching.")
PY

python -m verl.trainer.main_ppo_format \
    "hydra.run.dir=logs/grpo_runs/$EXPERIMENT_NAME" \
    data.train_files=data/multitq_search2/train_multiple.parquet \
    data.val_files=data/multitq_search2/test_multiple.parquet \
    data.train_data_num=null \
    data.val_data_num=24 \
    data.train_batch_size=60 \
    data.val_batch_size=24 \
    data.max_prompt_length=5200 \
    data.max_response_length=1000 \
    data.max_start_length=2000 \
    data.max_obs_length=1400 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    "actor_rollout_ref.model.path=$BASE_MODEL" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.grad_clip=5.0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=30 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=dummy_hf \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.actor.state_masking=true \
    algorithm.no_think_rl=false \
    'trainer.logger=[console]' \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.project_name=TempR1-Qwen15B \
    "trainer.experiment_name=$EXPERIMENT_NAME" \
    trainer.total_epochs=5 \
    trainer.total_training_steps=80 \
    "trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME" \
    max_turns=3 \
    retriever.url=http://127.0.0.1:8000/retrieve \
    retriever.topk=30 \
    2>&1 | tee logs/grpo_stage1_1p5b.log
