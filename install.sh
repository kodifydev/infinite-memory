#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${INFINITE_MEMORY_REPO_URL:-https://github.com/kodifydev/infinite-memory.git}"
INSTALL_ROOT="${INFINITE_MEMORY_INSTALL_ROOT:-$HOME/.local/share/infinite-memory}"
BIN_DIR="${INFINITE_MEMORY_BIN_DIR:-$HOME/.local/bin}"
VENV_DIR="$INSTALL_ROOT/venv"
BIN_PATH="$BIN_DIR/infinite-memory"

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required" >&2
  exit 1
fi

PYTHON_VERSION="$(python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

python3 - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("error: Python 3.11+ is required")
PY

mkdir -p "$INSTALL_ROOT" "$BIN_DIR"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install --upgrade "git+$REPO_URL"
ln -sfn "$VENV_DIR/bin/infinite-memory" "$BIN_PATH"

cat <<EOF
infinite-memory installed.

Python: $PYTHON_VERSION
Binary: $BIN_PATH
Package venv: $VENV_DIR

Try:
  infinite-memory --help
  infinite-memory init --path ~/memories
  infinite-memory index --force
  infinite-memory search "shipping label bug"
EOF

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *)
    cat <<EOF

Heads up: $BIN_DIR is not on PATH in this shell.
Add this to your shell profile:
  export PATH="$BIN_DIR:\$PATH"
EOF
    ;;
esac
