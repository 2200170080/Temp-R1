# Temp-R1 GitHub 自动同步

本仓库由服务器上的一个常驻进程自动推送到 GitHub，方向是**服务器 → GitHub 单向**。

## 事实

| 项 | 值 |
|---|---|
| 工作仓库 | `~/Temp-R1`（服务器：`ecs-instance-cdb4917c`） |
| 远端 | `backup` = `git@github.com:2200170080/Temp-R1.git` |
| 上游 | `origin` = `https://ghfast.top/https://github.com/zjukg/Temp-R1.git`（只读参考） |
| 密钥 | `~/.ssh/github_tempr1`（仓库部署密钥，已勾选 Allow write access） |
| 脚本 | `scripts/github-sync/autosync.sh` |
| 日志 | `scripts/github-sync/autosync.log`（已被 Git 忽略） |
| 会话 | tmux `sync` |

仓库级 `core.sshCommand` 固定为：

```
ssh -F /dev/null -i ~/.ssh/github_tempr1 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15
```

`-F /dev/null` 是必需的：全局 `~/.ssh/config` 里 `Host github.com` 指向 SSR 的部署密钥 `github_ssr`，
不绕过就会把所有推送认证成 `2200170080/SSR`，报 `Permission to 2200170080/Temp-R1.git denied to deploy key`。

## 工作方式

每 `INTERVAL` 秒（默认 60）：

1. `git add -A` 收集新增、修改、删除与改名。
2. 守卫：暂存区中超过 `MAX_MB`（默认 50）的文件用 `git reset` 撤出暂存区，工作区文件保留，日志记 `skipped(big)`。
3. 暂存区非空则提交 `auto: <时间> @ <主机名>`。
4. 有新提交则推 `HEAD:refs/heads/main`，失败重试 `RETRIES` 次、每次间隔 10 秒。

## 启动与停止

```bash
# 启动（同一工作区只允许一套）
tmux has-session -t '=sync' 2>/dev/null || tmux new-session -d -s sync -c "$HOME/Temp-R1" \
  bash -c 'set -o pipefail; ROOT="$HOME/Temp-R1" REMOTE=backup BRANCH=main INTERVAL=60 MAX_MB=50 RETRIES=3 bash "$HOME/Temp-R1/scripts/github-sync/autosync.sh" 2>&1 | tee -a "$HOME/Temp-R1/scripts/github-sync/autosync.log"'

# 观察
tmux attach -t sync          # Ctrl+b 再按 d 脱离
pgrep -af '[a]utosync.sh'

# 停止
tmux kill-session -t '=sync'
```

## 验证同步真的成功

会话存在、日志有 `pushed` 都不算数——推送失败时脚本只重试、不退出。要比较 SHA：

```bash
cd ~/Temp-R1
git rev-parse HEAD
git ls-remote backup refs/heads/main
```

两行 SHA 相同才算远端已收到该提交。

## 不同步的内容

见 `.gitignore`。核心是体积大或可重建的产物：

- `corpus/kg_index/`（约 2.8 G，FAISS 索引与 embedding memmap）
- `data/multitq_search2/`（约 499 M，GRPO 用的 6 个 parquet）
- `data/MultiTQ/train*.json`（各 131 M，超过 GitHub 单文件硬限）
- `data/MultiTQ/oversampled_for_sft*.json`、`outputs/`、`*.log`、`__pycache__/`、`*.pyc`

模型权重、vLLM 缓存等大对象也都不在仓库内。

## 已知限制

1. **单文件 100 MiB 是硬限**。`corpus/full_kg.jsonl` 为 98.40 MiB，只剩约 1.6 MiB 余量；该文件若被重新生成变大，推送会失败，且一旦进入历史就无法靠后来删除解决，只能重建历史。
2. **不自动处理分歧**。若在网页或另一台机器改了同一分支，下一轮 push 会 non-fast-forward 失败并持续重试；需要先停同步器，`git fetch backup main` 后人工核对合并。
3. **工作区共享**。`/root/user/...` 可能是共享盘，同一工作区只能有一节点运行同步器，否则会争抢 `.git/index.lock`。
4. **历史是重建的单条初始提交**。原克隆为 `--depth 1` 浅仓库，浅仓库不允许推送，因此用 `--orphan` 建了干净历史；上游 commit 记录不在本仓库内，需要对比时通过 `origin` 拉取。
