#!/usr/bin/env bash
set -euo pipefail

INSTALL_ROOT="${INFINITE_MEMORY_INSTALL_ROOT:-$HOME/.local/share/infinite-memory}"
BIN_DIR="${INFINITE_MEMORY_BIN_DIR:-$HOME/.local/bin}"
BIN_PATH="$BIN_DIR/infinite-memory"
PURGE=0

if [[ "${1:-}" == "--purge" ]]; then
  PURGE=1
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: ./uninstall.sh [--purge]

Removes the installed infinite-memory binary and package virtualenv.
With --purge, also removes user config and index data.
EOF
  exit 0
elif [[ -n "${1:-}" ]]; then
  echo "error: unknown argument: $1" >&2
  exit 2
fi

rm -f "$BIN_PATH"
rm -rf "$INSTALL_ROOT/venv"

if [[ $PURGE -eq 1 ]]; then
  CONFIG_HOME="${XDG_CONFIG_HOME:-$HOME/.config}"
  DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
  rm -rf "$CONFIG_HOME/infinite-memory" "$DATA_HOME/infinite-memory"
  echo "infinite-memory uninstalled and config/data purged."
else
  cat <<EOF
infinite-memory uninstalled.

Kept user config/data:
  ${XDG_CONFIG_HOME:-$HOME/.config}/infinite-memory
  ${XDG_DATA_HOME:-$HOME/.local/share}/infinite-memory

To remove those too, run:
  ./uninstall.sh --purge
EOF
fi
