#!/usr/bin/env bash
set -Eeuo pipefail

# One-command deploy for production:
# 1) push local main to origin
# 2) run server update script via SSH
#
# Usage:
#   scripts/deploy-prod.sh
#   scripts/deploy-prod.sh --host root@88.218.122.81 --branch main

HOST="root@88.218.122.81"
BRANCH="main"
REMOTE_PATH="/opt/telegram-context-bot"
SERVICE="telegram-context-bot"

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy-prod.sh [options]

Options:
  --host      SSH target (default: root@88.218.122.81)
  --branch    Git branch to push/deploy (default: main)
  --path      Bot path on server (default: /opt/telegram-context-bot)
  --service   systemd service name (default: telegram-context-bot)
  --help      Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --branch)
      BRANCH="${2:-}"
      shift 2
      ;;
    --path)
      REMOTE_PATH="${2:-}"
      shift 2
      ;;
    --service)
      SERVICE="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[deploy-prod] pushing branch '$BRANCH' to origin"
git push origin "$BRANCH"

echo "[deploy-prod] updating server '$HOST'"
"$ROOT_DIR/scripts/update-via-ssh.sh" \
  --host "$HOST" \
  --branch "$BRANCH" \
  --path "$REMOTE_PATH" \
  --service "$SERVICE"

echo "[deploy-prod] done"
