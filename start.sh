#!/usr/bin/env bash
# Start the server, then open the display page full screen on the window panel.
#   ./start.sh          real generation
#   ./start.sh --mock   fake images (to try the controls)
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

python run.py "$@" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT INT TERM

for _ in $(seq 1 30); do
  curl -s -o /dev/null http://localhost:8765/api/state && break
  sleep 1
done
./start-display.sh || echo "Couldn't open the display automatically; open http://localhost:8765/display on the panel yourself."

# keep the Mac awake while this runs (display may sleep; the panel is driven by the second screen)
caffeinate -i -w $SERVER
