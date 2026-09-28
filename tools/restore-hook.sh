#!/bin/bash
# Undo tools/install-hook.sh: put the user's original Alphabots.dll back and disable the harness.
set -euo pipefail

R="$(cd "$(dirname "$0")/.." && pwd)"
HOST="$HOME/Library/Application Support/com.mosimulator.mosimulator/Mods/China Modpack/Alphabots.dll"
BACKUP="$R/backups/Alphabots.dll.pre-mosimrl"

if pgrep -x MoSimulator >/dev/null; then
  echo "MoSimulator is running — quit it first." >&2
  exit 1
fi
[ -f "$BACKUP" ] || { echo "no backup at $BACKUP — nothing to restore" >&2; exit 1; }
cp "$BACKUP" "$HOST"
rm -f "$R/run/ENABLE" "$HOME/.mosimrl_run"
echo "restored: $HOST ($(md5 -q "$HOST"))"
