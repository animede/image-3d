#!/usr/bin/env bash
# Image-3D 起動スクリプト
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
  echo "venvが見つかりません。先に以下を実行してください:"
  echo "  python3 -m venv $VENV_DIR"
  echo "  $VENV_DIR/bin/pip install -r requirements.txt"
  exit 1
fi

export IMAGE3D_GENERATOR="${IMAGE3D_GENERATOR:-mock}"
export IMAGE3D_HOST="${IMAGE3D_HOST:-127.0.0.1}"
export IMAGE3D_PORT="${IMAGE3D_PORT:-8000}"

exec "$VENV_DIR/bin/uvicorn" server.main:app \
  --host "$IMAGE3D_HOST" \
  --port "$IMAGE3D_PORT" \
  "$@"
