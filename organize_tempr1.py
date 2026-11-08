#!/usr/bin/env python3
"""Preview/apply a conservative Temp-R1 layout cleanup. Python 3.9+, stdlib only.

Does not install packages, delete caches, stop services, commit, or push.
Stop this project's autosync before --apply or --undo.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime


LAYOUT = """# Temp-R1 目录说明

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
"""

NAV = """<!-- tempr1-local-navigation -->
## 本机复现导航

- [目录说明](docs/PROJECT_LAYOUT.md)
- [部署记录（待实测核验）](docs/Temp-R1_部署手册.md)
- [环境快照](docs/environment/)
- [GitHub 同步说明](docs/GitHub同步说明.md)

下方保留上游项目说明；本机 1.5B 训练配置尚待验证。

<!-- /tempr1-local-navigation -->

"""

IGNORE = ["*.egg-info/", "/logs/", "/verl_checkpoints/", "/LLaMA-Factory/saves/", "/wandb/", "/runs/", ".env.*"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode if path.exists() else 0o644
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as f:
        temporary = Path(f.name)
        f.write(content)
    os.chmod(str(temporary), mode)
    os.replace(str(temporary), str(path))


def check_paths(root, names):
    for name in names:
        path = root / name
        if path.is_symlink() or root not in path.resolve().parents:
            raise RuntimeError("路径不在预期位置：" + name)


def check_sync(root):
    proc = Path("/proc")
    if not proc.is_dir():
        return
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            args = (entry / "cmdline").read_bytes().split(b"\0")
            for arg in args:
                arg = os.fsdecode(arg)
                if Path(arg).name == "autosync.sh":
                    script = Path(arg)
                    if not script.is_absolute():
                        script = (entry / "cwd").resolve() / script
                    if script.resolve() == root / "scripts/github-sync/autosync.sh":
                        raise RuntimeError("同步器仍在运行，PID=" + entry.name + "；请先停止本项目的 tmux sync 会话。")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue


def has_open_file(path):
    proc = Path("/proc")
    if not proc.is_dir() or not path.exists():
        return False
    target = path.stat()
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            for fd in (entry / "fd").iterdir():
                try:
                    s = fd.stat()
                    if (s.st_dev, s.st_ino) == (target.st_dev, target.st_ino):
                        return True
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return False


def make_plan(root):
    moves = []
    candidates = [("Temp-R1_部署手册.md", "docs/Temp-R1_部署手册.md")]
    candidates += [(p.name, "docs/environment/" + p.name) for p in sorted(root.glob("env_freeze*.txt"))]
    if (root / "build_index.log").exists():
        if has_open_file(root / "build_index.log"):
            print("保留仍被进程打开的 build_index.log")
        else:
            candidates.append(("build_index.log", "logs/build_index.log"))
    for src, dst in candidates:
        source, target = root / src, root / dst
        if source.is_symlink() or target.is_symlink():
            raise RuntimeError("目标涉及软链接，需单独核对：" + src)
        if source.exists():
            if target.exists():
                raise RuntimeError("目标已存在，停止以避免覆盖：" + dst)
            moves.append({"src": src, "dst": dst, "sha256": digest(source)})
    if not (root / "Temp-R1_部署手册.md").is_file() and not (root / "docs/Temp-R1_部署手册.md").is_file():
        raise RuntimeError("未找到部署记录，请核对项目根目录。")

    edits = {}
    old = (root / ".gitignore").read_text(encoding="utf-8")
    missing = [line for line in IGNORE if line not in old.splitlines()]
    if missing:
        edits[".gitignore"] = old.rstrip() + "\n\n# Local installation metadata and experiment artifacts\n" + "\n".join(missing) + "\n"
    readme = (root / "README.md").read_text(encoding="utf-8")
    if "<!-- tempr1-local-navigation -->" not in readme:
        edits["README.md"] = NAV + readme.replace("](Temp-R1_部署手册.md)", "](docs/Temp-R1_部署手册.md)")
    layout_path = root / "docs/PROJECT_LAYOUT.md"
    if layout_path.exists() and layout_path.read_text(encoding="utf-8") != LAYOUT:
        raise RuntimeError("docs/PROJECT_LAYOUT.md 已有其他内容，停止以避免覆盖。")
    if not layout_path.exists():
        edits["docs/PROJECT_LAYOUT.md"] = LAYOUT
    sync_path = root / "docs/GitHub同步说明.md"
    if sync_path.exists():
        sync_doc = sync_path.read_text(encoding="utf-8")
        fixed = sync_doc.replace("、`outputs/`、`*.log`", "、`*.log`")
        note = "\n\n注：`outputs/` 中已有 SFT 轨迹已纳入 Git 跟踪；日志及模型产物按 `.gitignore` 排除。\n"
        if fixed != sync_doc and "outputs/` 中已有 SFT" not in fixed:
            fixed = fixed.rstrip() + note
        if fixed != sync_doc:
            edits["docs/GitHub同步说明.md"] = fixed
    return moves, edits


def apply_plan(root, moves, edits):
    check_sync(root)
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=str(root)).returncode:
        raise RuntimeError("当前存在未提交的已跟踪文件改动，请先贴回 git status --short，避免混入正在进行的修改。")
    backup = root.parent / (root.name + "-layout-backup-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    backup.mkdir()
    manifest = {"root": str(root), "moves": [], "edits": []}
    manifest["head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    def save():
        write_file(backup / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    save()
    print("恢复资料：", backup, flush=True)
    for path, content in edits.items():
        original = root / path
        existed = original.exists()
        if existed:
            copy = backup / "original" / path
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(original), str(copy))
        encoded = content.encode("utf-8")
        manifest["edits"].append({"path": path, "existed": existed, "sha256": hashlib.sha256(encoded).hexdigest()})
        save()
        write_file(original, encoded)
    for item in moves:
        src, dst = root / item["src"], root / item["dst"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or digest(src) != item["sha256"]:
            raise RuntimeError("移动前文件发生变化，停止：" + item["src"])
        manifest["moves"].append(item)
        save()
        src.rename(dst)
    manifest["complete"] = True
    save()
    print("整理完成；检索服务、模型配置、数据与索引未作修改。")
    print("回退命令：python3 " + str(Path(__file__).resolve()) + " --undo " + str(backup))
    subprocess.run(["git", "-c", "core.quotePath=false", "diff", "--stat"], cwd=str(root))
    subprocess.run(["git", "-c", "core.quotePath=false", "status", "--short"], cwd=str(root))


def undo(backup):
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    root = Path(manifest["root"]).resolve()
    check_sync(root)
    names = [item["path"] for item in manifest["edits"]]
    for item in manifest["moves"]:
        names.extend([item["src"], item["dst"]])
    check_paths(root, names)
    pending_moves, pending_edits = [], []
    for item in manifest["moves"]:
        src, dst = root / item["src"], root / item["dst"]
        if dst.exists() and not src.exists() and digest(dst) == item["sha256"]:
            pending_moves.append(item)
        elif not (src.exists() and not dst.exists() and digest(src) == item["sha256"]):
            raise RuntimeError("文件已有后续变化，未执行回退：" + item["src"])
    for item in manifest["edits"]:
        current = root / item["path"]
        original = backup / "original" / item["path"]
        if current.is_file() and digest(current) == item["sha256"]:
            pending_edits.append(item)
        elif item["existed"] and current.is_file() and original.is_file() and digest(current) == digest(original):
            continue
        elif not item["existed"] and not current.exists():
            continue
        else:
            raise RuntimeError("文件已有后续变化，未执行回退：" + item["path"])
    for item in reversed(pending_moves):
        (root / item["dst"]).rename(root / item["src"])
    for item in reversed(pending_edits):
        target = root / item["path"]
        if item["existed"]:
            shutil.copy2(str(backup / "original" / item["path"]), str(target))
        else:
            target.unlink()
    print("已回退本次文件修改；未修改 Git 历史，新增空目录和外部备份保留。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.home() / "Temp-R1")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--undo", type=Path, metavar="BACKUP_DIRECTORY")
    args = parser.parse_args()
    if args.undo:
        undo(args.undo.expanduser().resolve())
        return
    root = args.root.expanduser().resolve()
    if not (root / ".git").exists() or not (root / "verl").is_dir() or not (root / "LLaMA-Factory").is_dir():
        raise RuntimeError("指定目录不符合 Temp-R1 仓库结构：" + str(root))
    # Prevent writes through existing directory/file aliases.
    check_paths(root, ["docs", "docs/environment", "logs", ".gitignore", "README.md", "docs/PROJECT_LAYOUT.md", "docs/GitHub同步说明.md"])
    moves, edits = make_plan(root)
    print("项目：", root)
    for item in moves:
        print("移动：", item["src"], "->", item["dst"])
    for path in edits:
        print("更新：", path)
    print("保留：retrieval.log、数据/索引/轨迹、框架源码、安装元数据及 Python 缓存。")
    if args.apply:
        if not moves and not edits:
            print("已经整理，无需重复修改。")
        else:
            apply_plan(root, moves, edits)
    else:
        print("当前为预览，未修改任何文件；停止本项目自动同步后添加 --apply 执行。")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print("停止：" + str(error), file=sys.stderr)
        sys.exit(1)
