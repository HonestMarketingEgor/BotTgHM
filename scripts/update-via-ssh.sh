#!/usr/bin/env bash
set -Eeuo pipefail

# Simple server update over SSH:
# - connects to server
# - pulls latest code from git
# - installs requirements
# - restarts systemd service

HOST=""
REMOTE_PATH="/opt/telegram-context-bot"
BRANCH="main"
SERVICE="telegram-context-bot"
PYTHON_BIN="python3"

usage() {
  cat <<'EOF'
Usage:
  scripts/update-via-ssh.sh --host <user@server> [options]

Required:
  --host         SSH target, e.g. root@88.218.122.81

Optional:
  --path         Bot directory on server (default: /opt/telegram-context-bot)
  --branch       Git branch to deploy (default: main)
  --service      Systemd service name (default: telegram-context-bot)
  --python       Python binary for venv (default: python3)
  --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --path)
      REMOTE_PATH="${2:-}"
      shift 2
      ;;
    --branch)
      BRANCH="${2:-}"
      shift 2
      ;;
    --service)
      SERVICE="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_BIN="${2:-}"
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

if [[ -z "$HOST" ]]; then
  echo "Error: --host is required." >&2
  usage
  exit 1
fi

ssh "$HOST" "BRANCH='$BRANCH' REMOTE_PATH='$REMOTE_PATH' SERVICE='$SERVICE' PYTHON_BIN='$PYTHON_BIN' bash -s" <<'EOF'
set -Eeuo pipefail

cd "$REMOTE_PATH"

echo "[remote] update git branch: $BRANCH"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

if [[ ! -d ".venv" ]]; then
  echo "[remote] creating .venv"
  "$PYTHON_BIN" -m venv .venv
fi

echo "[remote] install dependencies"
.venv/bin/pip install --upgrade pip >/dev/null
.venv/bin/pip install -r requirements.txt

echo "[remote] restart service: $SERVICE"
systemctl restart "$SERVICE"
systemctl status "$SERVICE" --no-pager -l | sed -n '1,20p'

echo "[remote] done"
EOF
