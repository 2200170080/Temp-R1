#!/usr/bin/env bash
# Temp-R1: 服务器 -> GitHub 单向自动同步
set -uo pipefail

ROOT="${ROOT:-$HOME/Temp-R1}"
REMOTE="${REMOTE:-backup}"
BRANCH="${BRANCH:-main}"
INTERVAL="${INTERVAL:-60}"
MAX_MB="${MAX_MB:-50}"
RETRIES="${RETRIES:-3}"
KEY="${KEY:-$HOME/.ssh/github_tempr1}"
AUTHOR_NAME="${AUTHOR_NAME:-Temp-R1 Autosync}"
AUTHOR_EMAIL="${AUTHOR_EMAIL:-autosync@localhost}"

cd "$ROOT" || { echo "ROOT 不存在: $ROOT"; exit 1; }
export GIT_SSH_COMMAND="ssh -i $KEY -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"

log() { echo "$(date '+%F %T') $*"; }
log "autosync 启动 root=$ROOT remote=$REMOTE branch=$BRANCH interval=${INTERVAL}s max=${MAX_MB}MB"
NEED_PUSH=0

while true; do
  git add -A 2>/dev/null

  # 守卫：撤出超过 MAX_MB 的文件，保留工作区文件
  while IFS= read -r -d '' f; do
    [ -f "$f" ] || continue
    sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -gt $((MAX_MB * 1024 * 1024)) ]; then
      git reset -q -- "$f"
      log "skipped(big): $f $((sz / 1024 / 1024))MB"
    fi
  done < <(git diff --cached --name-only -z)

  if ! git diff --cached --quiet; then
    if git -c user.name="$AUTHOR_NAME" -c user.email="$AUTHOR_EMAIL" \
         commit -q -m "auto: $(date '+%F %T') @ $(hostname)"; then
      NEED_PUSH=1
      log "committed $(git log -1 --format=%h)"
    fi
  fi

  if [ "$NEED_PUSH" = "1" ]; then
    for i in $(seq 1 "$RETRIES"); do
      if git push "$REMOTE" "HEAD:refs/heads/$BRANCH" 2>&1; then
        NEED_PUSH=0
        log "pushed $(git rev-parse --short HEAD)"
        break
      fi
      log "push 失败 第${i}次，10s 后重试"
      sleep 10
    done
  fi

  sleep "$INTERVAL"
done
