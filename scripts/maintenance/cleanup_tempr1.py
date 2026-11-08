#!/usr/bin/env python3
"""整理 Temp-R1 的已知旧产物；默认只预览，--apply 才修改本机文件。

仅删除本次复现已确认的旧检查点、测试输出及历史测试日志/运行配置。
不启动训练，不停止服务，不提交 Git，不访问网络，不扫描 /proc/*/fd。
兼容服务器 Python 3.9，仅使用标准库。
"""

import argparse
from dataclasses import dataclass, field
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile


SFT = "LLaMA-Factory/saves/qwen2.5-1.5b-it/full/sft"
OLD_TENSORBOARD_RUN = SFT + "/runs/Nov08_12-45-37_ecs-instance-cdb4917c"
OLD_HYDRA_RUNS = tuple(
    prefix + "/" + time
    for prefix in ("logs/grpo_runs/2026-11-08", "outputs/2026-11-08")
    for time in ("14-46-41", "15-02-24", "15-13-15")
)
DELETE_DIRECTORY_ALLOWLIST = (
    SFT + "/checkpoint-50",
    SFT + "/checkpoint-100",
    "LLaMA-Factory/saves/qwen2.5-1.5b-it/full/sft-smoke-20261108-114928",
    "verl_checkpoints/qwen1p5b-stage1-smoke-20261108-150209",
) + OLD_HYDRA_RUNS
DELETE_FILE_ALLOWLIST = (
    "logs/build_index.log",
    "logs/grpo_preflight.log",
    "logs/grpo_memory_fix.log",
    "logs/sft_qwen_1p5b_smoke.log",
    "logs/grpo_stage1_1p5b_smoke.log",
    OLD_TENSORBOARD_RUN + "/events.out.tfevents.1794113172.ecs-instance-cdb4917c.1456972.0",
)
DELETE_ALLOWLIST = DELETE_DIRECTORY_ALLOWLIST + DELETE_FILE_ALLOWLIST
MOVES = (
    ("scripts/train_grpo_stage1_1p5b.sh",
     "scripts/multitq/train_grpo_stage1_1p5b.sh"),
    ("scripts/train_grpo_stage1_1p5b_smoke.sh",
     "scripts/multitq/train_grpo_stage1_1p5b_smoke.sh"),
    ("scripts/fix_grpo_response_logits.py",
     "scripts/maintenance/fix_grpo_response_logits.py"),
)
NAV_START = "<!-- tempr1-local-navigation -->"
NAV_END = "<!-- /tempr1-local-navigation -->"
NAV = """<!-- tempr1-local-navigation -->
## 当前 1.5B 复现入口

- [目录与产物说明](docs/PROJECT_LAYOUT.md)
- [历史部署记录](docs/Temp-R1_部署手册.md)
- [环境快照](docs/environment/) · [GitHub 同步说明](docs/GitHub同步说明.md)
- SFT：\x60scripts/multitq/train_sft_qwen_1p5b.sh\x60
- GRPO Stage1：\x60scripts/multitq/train_grpo_stage1_1p5b.sh\x60
- GRPO 两步测试：\x60scripts/multitq/train_grpo_stage1_1p5b_smoke.sh\x60

当前状态（2026-11-08）：正式 SFT 已完成，GRPO 两步测试通过；
正式 GRPO Stage1 第一步反向传播显存不足，现按用户要求暂停，尚未完成。
训练继续使用已有 \x60tempr1\x60 环境；现有检索服务保持原部署。
下方为上游原说明，其中多卡训练入口、环境和模型设置需按本机配置核对。
Stage2 与正式评测尚未完成；目录整理不会启动训练。

<!-- /tempr1-local-navigation -->"""
LAYOUT = """# Temp-R1 目录与产物说明

## 当前进度

正式 Qwen2.5-1.5B SFT 已完成，GRPO 两步测试通过。
正式 Stage1 在第一步反向传播时显存不足，现已暂停；Stage2 和正式评测尚未完成。
本次整理只处理目录和旧产物，不改变训练参数、模型权重或现有检索服务。

## 常用入口

| 用途 | 路径 |
|---|---|
| 1.5B SFT | \x60scripts/multitq/train_sft_qwen_1p5b.sh\x60 |
| 1.5B GRPO Stage1 | \x60scripts/multitq/train_grpo_stage1_1p5b.sh\x60 |
| 1.5B GRPO 两步测试 | \x60scripts/multitq/train_grpo_stage1_1p5b_smoke.sh\x60 |
| 已安装的 GRPO 显存补丁工具 | \x60scripts/maintenance/fix_grpo_response_logits.py\x60 |
| 本次整理工具（默认预览） | \x60scripts/maintenance/cleanup_tempr1.py\x60 |
| 目录整理旧工具 | \x60scripts/maintenance/organize_tempr1.py\x60 |
| 检索服务 | \x60scripts/retrieval/retrieval_launch.sh\x60 |
| GitHub 自动同步 | \x60scripts/github-sync/autosync.sh\x60 |

上述训练入口仅作定位。本次不运行训练；原作者其他脚本继续保留供参考。

## 目录职责

\x60\x60\x60text
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
\x60\x60\x60

保留 editable 安装的框架目录位置。正式权重和数据不为美化目录而搬迁。
检索服务正在使用的 \x60logs/retrieval.log\x60 与同步器的
\x60scripts/github-sync/autosync.log\x60 继续保留原位。

## 保留与清理边界

- 保留最终 SFT：\x60LLaMA-Factory/saves/qwen2.5-1.5b-it/full/sft/\x60
  中的独立权重、config、tokenizer、训练结果及 loss 图。
- 旧模型清理范围为已列明的 \x60checkpoint-50/\x60、\x60checkpoint-100/\x60、
  \x60sft-smoke-20261108-114928/\x60 和
  \x60qwen1p5b-stage1-smoke-20261108-150209/\x60。
  删除前检查最终 SFT 的配置、tokenizer 和 safetensors 文件结构。
- 同时清理本次建索引、预检、显存补丁与 SFT/GRPO 测试日志，以及
  \x60logs/grpo_runs/2026-11-08/\x60 下的
  \x6014-46-41/\x60、\x6015-02-24/\x60、\x6015-13-15/\x60 历史配置目录。
  保留正式 \x60sft_qwen_1p5b.log\x60 与最近的 \x60grpo_stage1_1p5b.log\x60。
- 删除已结束的 SFT TensorBoard 事件文件及其空目录；
  最终模型目录中的训练指标、\x60trainer_log.jsonl\x60 和 loss 图继续保留。
- 删除旧 SFT 检查点也会删除其中的优化器状态，不能再从第 50/100 步精确续训；
  独立最终 SFT 权重继续用于后续 GRPO 和推理。
- 保留所有未明确列入删除清单的模型目录，尤其正式 GRPO 的 \x60actor/global_step_*\x60。
- 保留作者轨迹、原始/加工数据、索引、关键成功/失败日志、安装元数据和缓存。
- 当前 1.5B GRPO 入口的新配置写到 \x60logs/grpo_runs/<experiment_name>/\x60。
  三套已结束的历史运行配置会实际删除，不再转存到归档目录。
- 已核实的阶段结论：正式 SFT 完成 100 步、2 个 epoch，平均训练 loss 约
  0.14289；响应 logits 的显存修复已通过 CPU 输出/梯度等价检查及两步
  GRPO 测试；正式 Stage1 在首步 \x60loss.backward()\x60 显存不足，
  当时剩余 1.29 GiB、还需分配 2.84 GiB。两步测试不代表正式评测结果。

## 整理后的检查

整理工具默认仅输出计划；\x60--apply\x60 会执行已列明的本机操作。
应先暂停本项目的 GitHub \x60sync\x60 会话；工具遇到仍在运行的训练或同步进程会停止。
完成后核对 \x60git diff --stat\x60、最终 SFT 和检索服务，再恢复同步。
本次操作不会提交、推送或启动实验。
"""


@dataclass
class Plan:
    root: Path
    deletes: list = field(default_factory=list)
    moves: list = field(default_factory=list)
    writes: dict = field(default_factory=dict)
    protected: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    reclaim_bytes: int = 0


def inside(root, relative):
    """不先跟随符号链接；逐级检查后才验证最终解析路径。"""
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError("不是项目内的相对路径: " + str(relative))
    path = root
    for part in rel.parts:
        path = path / part
        try:
            info = path.lstat()
        except FileNotFoundError:
            info = None
        reparse = info is not None and (
            getattr(info, "st_file_attributes", 0) &
            getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
        if path.is_symlink() or (
            hasattr(path, "is_junction") and path.is_junction()
        ) or reparse:
            raise RuntimeError("拒绝处理符号链接/junction: " + str(path))
    resolved = path.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError("路径超出项目: " + str(path))
    return path


def tree_bytes(root, relative):
    base = inside(root, relative)
    if not base.is_dir():
        raise RuntimeError("删除候选不是普通目录: " + str(base))
    total = 0
    for current, dirs, files in os.walk(str(base), followlinks=False, onerror=walk_error):
        for name in dirs + files:
            item = Path(current) / name
            inside(root, item.relative_to(root))
            info = item.lstat()
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
            elif not stat.S_ISDIR(info.st_mode):
                raise RuntimeError("候选包含特殊文件，保留并停止: " + str(item))
    return total


def artifact_bytes(root, relative):
    path = inside(root, relative)
    if path.is_dir():
        return tree_bytes(root, relative)
    if path.is_file() and stat.S_ISREG(path.lstat().st_mode):
        return path.stat().st_size
    raise RuntimeError("清理候选不是普通文件或目录: " + str(path))


def check_delete_candidate(root, relative):
    path = inside(root, relative)
    if relative in DELETE_FILE_ALLOWLIST:
        if not path.is_file():
            raise RuntimeError("清理清单中的文件变成其他类型，保留并停止: " + str(path))
    elif relative in DELETE_DIRECTORY_ALLOWLIST:
        if not path.is_dir():
            raise RuntimeError("清理清单中的目录变成其他类型，保留并停止: " + str(path))
    else:
        raise RuntimeError("删除路径不在固定清单: " + relative)
    return artifact_bytes(root, relative)


def walk_error(error):
    raise error


def tree_stamp(root, relative):
    """只读记录候选路径、大小与修改时间，供执行前复查。"""
    base = inside(root, relative)
    if base.is_file():
        return signature(base)
    stamps = []
    for current, dirs, files in os.walk(str(base), followlinks=False, onerror=walk_error):
        for name in sorted(dirs + files):
            item = Path(current) / name
            inside(root, item.relative_to(root))
            info = item.lstat()
            stamps.append((
                item.relative_to(base).as_posix(), info.st_mode,
                info.st_size, info.st_mtime_ns,
            ))
    return tuple(sorted(stamps))


def signature(path):
    info = path.stat()
    if path.name.endswith(".safetensors"):
        with path.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise RuntimeError("safetensors 文件不完整: " + str(path))
            length = struct.unpack("<Q", raw)[0]
            if not 0 < length <= 32 * 1024 * 1024:
                raise RuntimeError("safetensors 头长度异常: " + str(path))
            header = stream.read(length)
        tensors = json.loads(header)
        offsets = [v["data_offsets"] for k, v in tensors.items() if k != "__metadata__"]
        if not offsets or 8 + length + max(v[1] for v in offsets) != info.st_size:
            raise RuntimeError("safetensors 文件长度与张量索引不符: " + str(path))
        for start, end in offsets:
            if not 0 <= start <= end <= info.st_size - 8 - length:
                raise RuntimeError("safetensors 张量边界异常: " + str(path))
        digest = hashlib.sha256(header).hexdigest()
    else:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return (info.st_size, info.st_mtime_ns, digest)


def check_project(root):
    if not inside(root, ".git").is_dir():
        raise RuntimeError("目标不是已部署的 Git 项目: " + str(root))
    for relative in (
        "verl/trainer/main_ppo_format.py",
        "LLaMA-Factory/src/llamafactory/train/tuner.py",
        "outputs/train_sft_oversampled_merged.jsonl",
        "outputs/gpt4o-1000-cron-new.jsonl",
    ):
        if not inside(root, relative).is_file():
            raise RuntimeError("缺少项目保护标记: " + relative)
    config = json.loads(inside(root, SFT + "/config.json").read_text(encoding="utf-8"))
    tokenizer = json.loads(
        inside(root, SFT + "/tokenizer_config.json").read_text(encoding="utf-8")
    )
    if config.get("model_type") != "qwen2" or not tokenizer.get("chat_template"):
        raise RuntimeError("最终 SFT 的 Qwen2 配置或对话模板不完整，停止整理。")
    weights = inside(root, SFT + "/model.safetensors")
    if not weights.is_file() or weights.stat().st_size < 1024 * 1024:
        raise RuntimeError("缺少独立最终 SFT 权重；禁止删除其检查点。")
    paths = [
        SFT + "/model.safetensors", SFT + "/config.json",
        SFT + "/tokenizer.json", SFT + "/tokenizer_config.json",
        "outputs/train_sft_oversampled_merged.jsonl",
        "outputs/gpt4o-1000-cron-new.jsonl",
    ]
    return {relative: signature(inside(root, relative)) for relative in paths}


def ensure_idle(root):
    if sys.platform != "linux":
        raise RuntimeError("--apply 仅供 Linux 服务器；本机可预览和仿真测试。")
    blockers = []
    try:
        entries = list(Path("/proc").iterdir())
    except OSError as error:
        raise RuntimeError("无法检查运行状态，未执行整理: " + str(error))
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = [
                value.decode("utf-8", "replace")
                for value in (entry / "cmdline").read_bytes().split(b"\0") if value
            ]
        except OSError as error:
            if error.errno in (errno.ENOENT, errno.ESRCH):
                continue
            raise RuntimeError("进程状态无法确认，未执行整理: " + str(error))
        args = " ".join(argv)
        # tmux 服务端可能长期保留首次启动命令，不能据此判定 sync 仍运行。
        if argv and Path(argv[0]).name in ("tmux", "pgrep", "ps", "grep"):
            continue
        training = any(
            value.startswith("verl.trainer.main_") or
            value == "llamafactory.cli" or
            value.endswith("/llamafactory/launcher.py") or
            value.startswith(("ray::WorkerDict", "ray::main_task"))
            for value in argv
        ) or (
            any(Path(value).name == "llamafactory-cli" for value in argv) and
            "train" in argv
        )
        possible_sync = "autosync.sh" in args
        if not (training or possible_sync):
            continue
        try:
            cwd = (entry / "cwd").resolve(strict=True)
        except OSError as error:
            if error.errno in (errno.ENOENT, errno.ESRCH):
                continue
            raise RuntimeError("相关进程工作目录无法确认，未执行整理: " + str(error))
        in_project = cwd == root or root in cwd.parents or str(root) + "/" in args
        target = root / "scripts/github-sync/autosync.sh"
        sync = possible_sync and (
            str(target) in args or
            any(
                (cwd / value).resolve() == target
                for value in argv if value.endswith("autosync.sh")
            ) or
            (in_project and "scripts/github-sync/autosync.sh" in args)
        )
        training = training and in_project
        if training or sync:
            blockers.append(entry.name + ": " + args[:240])
    if (root / ".git/index.lock").exists():
        blockers.append(".git/index.lock 存在（不会自动删除）")
    if blockers:
        raise RuntimeError("请先停止本项目的训练/同步，其他服务保持运行:\n" + "\n".join(blockers))


def check_hydra_directory(path):
    for run in path.iterdir():
        if not run.is_dir() or not re.fullmatch(r"\d{2}-\d{2}-\d{2}", run.name):
            raise RuntimeError("outputs 日期目录有非预期内容，保留并停止: " + str(run))
        check_hydra_run(run)


def check_hydra_run(run):
    files = {item.relative_to(run).as_posix() for item in run.rglob("*") if item.is_file()}
    expected = {".hydra/config.yaml", ".hydra/hydra.yaml", ".hydra/overrides.yaml"}
    # Hydra 默认还会创建该日志；旧 Git 忽略规则使其未出现在远端文件树。
    optional = {"main_ppo_format.log"}
    missing = expected - files
    extra = files - expected - optional
    if missing or extra:
        raise RuntimeError(
            "运行目录有未识别内容或缺少配置，保留并停止: {}\n缺少: {}\n额外: {}".format(
                run, ", ".join(sorted(missing)) or "无",
                ", ".join(sorted(extra)) or "无",
            )
        )


def add_write(plan, relative, content):
    path = inside(plan.root, relative)
    before = path.read_bytes() if path.is_file() else None
    after = content.encode("utf-8")
    if before != after:
        plan.writes[relative] = (before, after)


def build_plan(root):
    root = root.expanduser().resolve()
    plan = Plan(root=root, protected=check_project(root))
    for relative in DELETE_ALLOWLIST:
        path = inside(root, relative)
        if path.exists():
            plan.reclaim_bytes += check_delete_candidate(root, relative)
            if relative in OLD_HYDRA_RUNS:
                check_hydra_run(path)
            plan.deletes.append(relative)
            plan.sources[relative] = tree_stamp(root, relative)
    for source, destination in MOVES:
        src, dst = inside(root, source), inside(root, destination)
        if src.exists() and dst.exists():
            raise RuntimeError("移动目标已存在，不覆盖: " + destination)
        if src.exists():
            if src.is_dir():
                tree_bytes(root, source)
                check_hydra_directory(src)
            else:
                if not src.is_file():
                    raise RuntimeError("移动源不是普通文件: " + source)
                signature(src)
            plan.sources[source] = tree_stamp(root, source)
            plan.moves.append((source, destination))

    readme = inside(root, "README.md").read_text(encoding="utf-8")
    if readme.count(NAV_START) != 1 or readme.count(NAV_END) != 1:
        raise RuntimeError("README 导航标记与预期不同，停止并保留原文件。")
    start, end = readme.index(NAV_START), readme.index(NAV_END) + len(NAV_END)
    readme = readme[:start] + NAV + readme[end:]
    add_write(plan, "README.md", readme)
    add_write(plan, "docs/PROJECT_LAYOUT.md", LAYOUT)

    for source, destination in MOVES[:2]:
        path = inside(root, source if (root / source).is_file() else destination)
        if not path.is_file():
            raise RuntimeError("找不到需要整理的 1.5B 训练入口: " + source)
        text = path.read_text(encoding="utf-8")
        if 'cd "$HOME/Temp-R1"' not in text:
            raise RuntimeError("训练入口工作目录与已核对版本不同: " + str(path))
        command = "python -m verl.trainer.main_ppo_format \\\n"
        option = '    "hydra.run.dir=logs/grpo_runs/$EXPERIMENT_NAME" \\\n'
        if option not in text:
            if text.count(command) != 1 or "hydra.run.dir=" in text:
                raise RuntimeError("训练入口与预期不同，停止: " + str(path))
            text = text.replace(command, command + option, 1)
        before = path.read_bytes()
        after = text.encode("utf-8")
        if before != after:
            # 文件先改名，再以原内容作并发检查。
            plan.writes[destination] = (before, after)

    # 只更新明确的旧路径引用；历史日志和模型源码均不改写。
    for path in sorted((root / "docs").rglob("*.md")):
        relative = path.relative_to(root).as_posix()
        if relative == "docs/PROJECT_LAYOUT.md":
            continue
        inside(root, relative)
        text = path.read_text(encoding="utf-8")
        for source, destination in MOVES[:3]:
            text = text.replace(source, destination)
        add_write(plan, relative, text)

    ignore = inside(root, ".gitignore").read_text(encoding="utf-8")
    block = (
        "\n# Hydra configs live in logs/grpo_runs; legacy dated outputs are runtime files\n"
        "/outputs/20??-??-??/\n"
        "!/logs/\n"
        "!/logs/grpo_runs/\n"
        "!/logs/grpo_runs/**\n"
    )
    if "# Hydra configs live in logs/grpo_runs;" not in ignore:
        add_write(plan, ".gitignore", ignore.rstrip() + "\n" + block)
    return plan


def show_plan(plan):
    print("项目:", plan.root)
    print("保留: 最终 SFT、作者轨迹、数据、索引、正式 GRPO 产物、关键日志和源码修复")
    for relative in plan.deletes:
        size = check_delete_candidate(plan.root, relative)
        print("删除: {} ({})".format(relative, format_bytes(size)))
    print("候选文件大小合计: {}（实际释放量受文件系统影响）".format(format_bytes(plan.reclaim_bytes)))
    for source, destination in plan.moves:
        print("移动:", source, "->", destination)
    for relative in plan.writes:
        print("更新:", relative)
    if any(relative.startswith(SFT + "/checkpoint-") for relative in plan.deletes):
        print("删除检查点将移除其优化器续训状态；独立最终 SFT 权重保留。")


def format_bytes(size):
    if size >= 1024 ** 3:
        return "{:.2f} GiB".format(size / (1024 ** 3))
    if size >= 1024 ** 2:
        return "{:.2f} MiB".format(size / (1024 ** 2))
    return "{:.1f} KiB".format(size / 1024)


def atomic_write(path, before, after):
    if (path.read_bytes() if path.exists() else None) != before:
        raise RuntimeError("文件在规划后发生变化，停止且不覆盖: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode if path.exists() else None
    descriptor, temporary = tempfile.mkstemp(prefix=".cleanup-", dir=str(path.parent))
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(after)
        if mode is not None:
            os.chmod(str(temporary), stat.S_IMODE(mode))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def execute_plan(plan, process_guard=ensure_idle):
    process_guard(plan.root)
    if check_project(plan.root) != plan.protected:
        raise RuntimeError("最终 SFT 或作者轨迹发生变化，未执行整理。")
    # 在第一次修改前重新生成全计划，拒绝所有源内容/路径变化。
    fresh = build_plan(plan.root)
    if fresh != plan:
        raise RuntimeError("项目在预览后发生变化，未执行整理，请重新预览。")
    for source, destination in plan.moves:
        src, dst = inside(plan.root, source), inside(plan.root, destination)
        if tree_stamp(plan.root, source) != plan.sources[source]:
            raise RuntimeError("移动源在检查后发生变化，停止: " + source)
        if dst.exists():
            raise RuntimeError("目标已出现，停止且不覆盖: " + destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        print("已移动:", source, "->", destination)
    for relative, (before, after) in plan.writes.items():
        atomic_write(inside(plan.root, relative), before, after)
        print("已更新:", relative)
    # 不可逆删除放在目录/文档更新之后；逐项再次核对根与白名单。
    process_guard(plan.root)
    for relative in plan.deletes:
        if relative not in DELETE_ALLOWLIST:
            raise RuntimeError("删除路径不在固定清单: " + relative)
        if check_project(plan.root) != plan.protected:
            raise RuntimeError("保护文件发生变化，停止删除。")
        path = inside(plan.root, relative)
        check_delete_candidate(plan.root, relative)
        if tree_stamp(plan.root, relative) != plan.sources[relative]:
            raise RuntimeError("候选在检查后发生变化，停止删除: " + relative)
        if path.is_dir():
            shutil.rmtree(str(path))
        else:
            path.unlink()
        print("已删除:", relative)
    # 只移除已知的空父目录，不递归删除其他内容。
    for relative in (
        "logs/grpo_runs/2026-11-08", "logs/grpo_runs", "outputs/2026-11-08",
        OLD_TENSORBOARD_RUN, SFT + "/runs",
    ):
        path = inside(plan.root, relative)
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
            print("已移除空目录:", relative)
    if check_project(plan.root) != plan.protected:
        raise RuntimeError("整理后保护文件不一致，请保留现场检查。")
    print("整理完成：最终 SFT 文件状态及作者轨迹校验一致；未启动训练或 Git 同步。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "Temp-R1")
    parser.add_argument("--apply", action="store_true", help="执行固定清单中的删除、移动和文档更新")
    args = parser.parse_args()
    plan = build_plan(args.root)
    show_plan(plan)
    if not args.apply:
        print("以上仅为预览，文件没有修改。执行整理需添加 --apply。")
        return
    execute_plan(plan)
    status = subprocess.run(
        ["git", "-C", str(plan.root), "status", "--short"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(status.stdout.rstrip())


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print("停止:", error, file=sys.stderr)
        sys.exit(1)
