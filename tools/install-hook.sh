#!/bin/bash
# Install the MoSimRL startup hook into the user's own mod DLL (China Modpack/Alphabots.dll).
# - keeps a pristine backup at backups/Alphabots.dll.pre-mosimrl (never overwritten once made)
# - always patches FROM that backup, so re-running is idempotent
# - undo with tools/restore-hook.sh
set -euo pipefail

if [ -d "$HOME/.dotnet" ]; then          # a user-local .NET SDK; otherwise use whatever `dotnet` is on PATH
  export DOTNET_ROOT="$HOME/.dotnet"
  export PATH="$HOME/.dotnet:$HOME/.dotnet/tools:$PATH"
fi
export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1

R="$(cd "$(dirname "$0")/.." && pwd)"
MODS="$HOME/Library/Application Support/com.mosimulator.mosimulator/Mods"
HOST="$MODS/China Modpack/Alphabots.dll"
BACKUP="$R/backups/Alphabots.dll.pre-mosimrl"
MANAGED="$HOME/Library/Application Support/Steam/steamapps/common/MoSimulator/MoSimulator.app/Contents/Resources/Data/Managed"
HARNESS="$R/harness/bin/MoSimRL.dll"
HOST_TYPE="Prefabs.Reefscape.Robots.Mods.China_Modpack._8810.Alphabots"

if pgrep -x MoSimulator >/dev/null; then
  echo "MoSimulator is running — quit it first (mods are only read at startup)." >&2
  exit 1
fi
[ -f "$HARNESS" ] || { echo "missing $HARNESS — build harness/MoSimRL first" >&2; exit 1; }

mkdir -p "$R/backups" "$R/run"
if [ ! -f "$BACKUP" ]; then
  cp "$HOST" "$BACKUP"
  echo "backup: $BACKUP ($(md5 -q "$BACKUP"))"
else
  echo "backup already present: $BACKUP ($(md5 -q "$BACKUP"))"
fi

TMP="$(mktemp -d)"
dotnet run --project "$R/tools/Injector" -c Release -- "$BACKUP" "$TMP/Alphabots.dll" "$MANAGED" "$HARNESS" "$HOST_TYPE"
cp "$TMP/Alphabots.dll" "$HOST"
rm -rf "$TMP"
echo "installed: $HOST ($(md5 -q "$HOST"))"
echo "$R/run" > "$HOME/.mosimrl_run"      # tells the harness where this clone's run/ (flag files, demos) is
echo "the hook is inert until $R/run/ENABLE exists"
