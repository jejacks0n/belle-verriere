#!/usr/bin/env bash
# Start the server and keep the Mac awake while it runs.
#   ./start.sh          real generation
#   ./start.sh --mock   fake images (to try the controls)
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

python run.py "$@" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT INT TERM

# keep the Mac awake while this runs (display may sleep; the panel is driven by the second screen)
caffeinate -i -w $SERVER
