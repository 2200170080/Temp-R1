#!/usr/bin/env bash
# Put this file at ~/Temp-R1/scripts/multitq/start_grpo_a6000.sh.
# Run on the chosen A6000: bash ~/Temp-R1/scripts/multitq/start_grpo_a6000.sh
# Uses the existing tempr1/retriever environments; never stops other processes.
set -eo pipefail

ROOT="$HOME/Temp-R1"
SELF="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
TRAIN="$ROOT/scripts/multitq/train_grpo_stage1_1p5b.sh"
PY="$HOME/miniconda3/envs/tempr1/bin/python"
SESSION=grpo-stage1-a6000
RETRIEVAL_SESSION=tempr1-retrieval

fail() { printf '停止：%s\n' "$*" >&2; exit 1; }

check_gpu() {
    # 可选门槛：启动前默认 43000；检索服务占住显存之后用更低的门槛二次校验。
    local minfree="${1:-43000}"
    local name total free
    IFS=, read -r name total free < <(
        nvidia-smi -i 0 --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    ) || fail "无法读取 GPU 0。"
    total="${total//[[:space:]]/}"
    free="${free//[[:space:]]/}"
    [[ "$total" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] || fail "显存查询结果无效。"
    [[ "$name" == *A6000* ]] && (( total >= 46000 )) || fail "当前 GPU 0 不是 48GB A6000：$name。"
    (( free >= minfree )) || fail "GPU 0 空闲显存只有 $free MiB，需要至少 $minfree MiB；请先手动释放自己的占用。"
    printf 'GPU：%s；总量 %s MiB；空闲 %s MiB（门槛 %s）\n' "$name" "$total" "$free" "$minfree"
}

check_training() {
    local processes
    processes="$(ps -u "$(id -u)" -o stat=,args=)" || fail "无法核对本机进程。"
    if printf '%s\n' "$processes" | awk '$1 !~ /^Z/ && /[v]erl[.]trainer[.]main_ppo_format|[r]ay::WorkerDict/ {found=1} END {exit !found}'; then
        fail "本机还有自己的 GRPO 训练或 Ray Worker；未重复启动。"
    fi
}

check_inputs() {
    local path
    command -v tmux >/dev/null || fail "找不到 tmux。"
    [[ -x "$PY" ]] || fail "找不到已有 tempr1 Python：$PY。"
    [[ -x "$HOME/miniconda3/envs/retriever/bin/python" ]] || fail "找不到已有 retriever Python。"
    for path in \
        "$TRAIN" \
        "$ROOT/scripts/retrieval/retrieval_launch.sh" \
        "$ROOT/LLaMA-Factory/saves/qwen2.5-1.5b-it/full/sft/model.safetensors" \
        "$ROOT/data/multitq_search2/train_multiple.parquet" \
        "$ROOT/data/multitq_search2/test_multiple.parquet" \
        "$ROOT/corpus/kg_index/e5_Flat.index" \
        "$ROOT/corpus/full_kg.jsonl"; do
        [[ -s "$path" ]] || fail "缺少文件或文件为空：$path。"
    done
    [[ "$(grep -c '^EXPERIMENT_NAME=' "$TRAIN")" == 1 ]] || fail "训练脚本的实验名称赋值已变化。"
    [[ "$(grep -Fc 'tee logs/grpo_stage1_1p5b.log' "$TRAIN")" == 1 ]] || fail "训练脚本的日志出口已变化。"
    [[ "$(grep -Fc 'retriever.url=http://127.0.0.1:8000/retrieve' "$TRAIN")" == 1 ]] || fail "训练脚本的本机检索地址已变化。"
}

cd -- "$ROOT" || fail "无法进入项目：$ROOT。"
check_inputs

if [[ "${1:-}" != --worker ]]; then
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
        fail "本机 tmux 会话 $SESSION 已存在；请勿重复启动。"
    fi
    check_training
    _parent_minfree=43000
    if tmux has-session -t "=$RETRIEVAL_SESSION" 2>/dev/null; then
        _parent_minfree="${MIN_FREE_WITH_RETRIEVAL:-39500}"
    fi
    check_gpu "$_parent_minfree"
    NODE="$(hostname)"
    NODE="${NODE//[^a-zA-Z0-9._-]/_}"
    RUN_TAG="$NODE-$(date +%Y%m%d-%H%M%S)-$$"
    BYPASS="${NO_PROXY:-},${no_proxy:-},127.0.0.1,localhost"
    mkdir -p logs
    tmux new-session -d -s "$SESSION" -c "$ROOT" \
        /usr/bin/env PATH="$PATH" PYTHONPATH="${PYTHONPATH:-}" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
        NO_PROXY="$BYPASS" no_proxy="$BYPASS" TEMPR1_RUN_TAG="$RUN_TAG" \
        /bin/bash "$SELF" --worker
    printf '已提交到本机 tmux：%s\n日志：%s/logs/grpo_stage1_1p5b_%s.log\n' "$SESSION" "$ROOT" "$RUN_TAG"
    printf "查看：tmux capture-pane -p -t '=%s:' -S -60\n" "$SESSION"
    exit 0
fi

[[ "${TEMPR1_RUN_TAG:-}" =~ ^[a-zA-Z0-9._-]+$ ]] || fail "内部运行标识无效。"
RUN_TAG="$TEMPR1_RUN_TAG"
RUN_LOG="$ROOT/logs/grpo_stage1_1p5b_$RUN_TAG.log"
RETRIEVAL_LOG="$ROOT/logs/retrieval_$RUN_TAG.log"
EXPERIMENT_NAME="qwen1p5b-stage1-$RUN_TAG"

worker_main() {
    printf '节点：%s\n实验：%s\n训练日志：%s\n' "$(hostname)" "$EXPERIMENT_NAME" "$RUN_LOG"
    check_training
    # 本机检索服务若已在跑（占 3~3.5 GiB），门槛要相应下调，否则永远过不了这一关。
    _minfree=43000
    if tmux has-session -t "=$RETRIEVAL_SESSION" 2>/dev/null; then
        _minfree="${MIN_FREE_WITH_RETRIEVAL:-39500}"
    fi
    check_gpu "$_minfree"

    "$PY" - "$ROOT" "$RETRIEVAL_SESSION" "$RETRIEVAL_LOG" <<'PY'
import errno
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import requests

root, session, logfile = sys.argv[1:]
url = "http://127.0.0.1:8000/retrieve"
client = requests.Session()
client.trust_env = False


def probe():
    response = client.post(url, json={"queries": ["China"], "topk": 1, "return_scores": True}, timeout=(2, 15))
    response.raise_for_status()
    hit = response.json()["result"][0][0]
    document = hit["document"]
    if document["id"] != "101702" or document["contents"] != "Peter_Pace Make_a_visit China on 2007-03-22" or not math.isfinite(float(hit["score"])):
        raise SystemExit("本机 8000 的检索内容不符，未启动训练。")


try:
    probe()
except requests.ConnectionError:
    with socket.socket() as sock:
        sock.settimeout(2)
        port_status = sock.connect_ex(("127.0.0.1", 8000))
        if port_status == 0:
            raise SystemExit("本机 8000 已被服务占用，但检索接口不可用；未启动其他服务。")
        if port_status != errno.ECONNREFUSED:
            raise SystemExit("无法确认本机 8000 端口空闲（错误码 " + str(port_status) + "），未启动其他服务。")
    exists = subprocess.run(["tmux", "has-session", "-t", "=" + session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if exists:
        print("本机检索会话已存在，等待它就绪。", flush=True)
    else:
        command = [
            "tmux", "new-session", "-d", "-s", session, "-c", root,
            "/usr/bin/env", "PATH=" + os.environ["PATH"],
            "PYTHONPATH=" + os.environ.get("PYTHONPATH", ""), "LD_LIBRARY_PATH=" + os.environ.get("LD_LIBRARY_PATH", ""),
            "NO_PROXY=" + os.environ.get("NO_PROXY", ""), "no_proxy=" + os.environ.get("no_proxy", ""),
            "/bin/bash", "-c", 'set -eo pipefail; exec /bin/bash "$1" >>"$2" 2>&1',
            "bash", str(Path(root) / "scripts/retrieval/retrieval_launch.sh"), logfile,
        ]
        subprocess.run(command, check=True)
        print("已在当前节点启动检索服务，日志：" + logfile, flush=True)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            probe()
            break
        except requests.ConnectionError:
            status = subprocess.run(["tmux", "has-session", "-t", "=" + session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if status.returncode != 0:
                raise SystemExit("检索启动会话已退出，请查看检索日志：" + logfile)
            time.sleep(2)
    else:
        raise SystemExit("180 秒内检索未就绪，未启动训练；本机检索会话保留供检查。")
else:
    print("复用当前节点已有的检索服务。", flush=True)

print("LOCAL_RETRIEVAL_OK：" + url, flush=True)
PY

    check_training
    # 检索服务已在本机 GPU 上占了约 3.5 GiB，二次校验用更低门槛（仍留 9 GiB 余量），
    # 否则本脚本永远无法在同一张卡上完成“起检索 + 起训练”。
    check_gpu "${MIN_FREE_AFTER_RETRIEVAL:-39500}"
    printf '检索已就绪，开始 GRPO Stage1。\n'
    sed \
        -e "s#^EXPERIMENT_NAME=.*#EXPERIMENT_NAME=\"$EXPERIMENT_NAME\"#" \
        -e 's#tee logs/grpo_stage1_1p5b\.log#tee /dev/null#' \
        "$TRAIN" | bash
}

worker_main 2>&1 | tee -a "$RUN_LOG"
