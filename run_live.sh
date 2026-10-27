#!/usr/bin/env bash
# Launch the excavator sim with hot-reloading controls (scripts/run_live.py).
# Edit scripts/live_controller.py or scripts/rocks.py while it runs; saves apply live.
#
# Usage: ./run_live.sh            (from anywhere)
# Set ISAACSIM=/path/to/isaacsim if Isaac Sim isn't installed at ~/isaacsim.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACSIM="${ISAACSIM:-$HOME/isaacsim}"

if [[ ! -x "$ISAACSIM/python.sh" ]]; then
    echo "Isaac Sim not found at $ISAACSIM (set ISAACSIM=/path/to/isaacsim)." >&2
    exit 1
fi

# Two Isaac Sim instances at once can freeze this machine (memory).
if pgrep -f "^$ISAACSIM/kit/python/bin/python3" >/dev/null; then
    echo "Isaac Sim is already running. Close it first:" >&2
    pgrep -af "^$ISAACSIM/kit/python/bin/python3" | cut -c1-120 >&2
    exit 1
fi

cd "$REPO"
exec "$ISAACSIM/python.sh" scripts/run_live.py "$@"
