#!/usr/bin/env bash
set -Eeuo pipefail

# Universal bot deploy via rsync + remote restart.
# Example:
#   scripts/deploy.sh \
#     --host user@your-server \
#     --path /opt/telegram-context-bot \
#     --restart "systemctl restart telegram-context-bot"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HOST=""
REMOTE_PATH=""
BRANCH="main"
PYTHON_BIN="python3"
RESTART_CMD=""

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy.sh --host <user@server> --path <remote/path> [options]

Required:
  --host        SSH target, e.g. ubuntu@203.0.113.10
  --path        Absolute path to bot directory on the server

Optional:
  --branch      Branch label used in logs (default: main)
  --python      Python binary on server (default: python3)
  --restart     Command to restart bot on server
                (default: systemctl restart telegram-context-bot || systemctl restart telegram-bot)
  --help        Show this help

Notes:
  - Keeps server .env as-is (does not overwrite local .env).
  - Creates/updates .venv and installs requirements.txt on server.
  - Requires rsync + ssh on local machine.
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
    --python)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --restart)
      RESTART_CMD="${2:-}"
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

if [[ -z "$HOST" || -z "$REMOTE_PATH" ]]; then
  echo "Error: --host and --path are required." >&2
  usage
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "Error: rsync is not installed locally." >&2
  exit 1
fi

if [[ -z "$RESTART_CMD" ]]; then
  RESTART_CMD='systemctl restart telegram-context-bot || systemctl restart telegram-bot'
fi

echo "[deploy] syncing project to $HOST:$REMOTE_PATH"
rsync -az --delete \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude ".DS_Store" \
  --exclude ".cursor/" \
  --exclude "data/" \
  --exclude ".env" \
  "$ROOT_DIR/" "$HOST:$REMOTE_PATH/"

echo "[deploy] applying update on server"
ssh "$HOST" "BRANCH='$BRANCH' PYTHON_BIN='$PYTHON_BIN' REMOTE_PATH='$REMOTE_PATH' RESTART_CMD='$RESTART_CMD' bash -s" <<'EOF'
set -Eeuo pipefail

cd "$REMOTE_PATH"
echo "[remote] deploy branch label: $BRANCH"

if [[ ! -d ".venv" ]]; then
  echo "[remote] creating virtual environment"
  "$PYTHON_BIN" -m venv .venv
fi

echo "[remote] installing dependencies"
.venv/bin/pip install --upgrade pip >/dev/null
.venv/bin/pip install -r requirements.txt

echo "[remote] restarting bot"
bash -lc "$RESTART_CMD"

echo "[remote] done"
EOF

echo "[deploy] success"
