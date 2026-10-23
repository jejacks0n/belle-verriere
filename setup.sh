#!/usr/bin/env bash
# One-time setup on the Mac. Safe to run again.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Python 3.10+ is needed (found $("$PY" --version)). Install it with: brew install python@3.12"
  exit 1
fi

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

if command -v ollama >/dev/null 2>&1; then
  if ! ollama list | grep -q "llama3.1:8b"; then
    echo "Pulling a small language model for prompt drift (about 5 GB)..."
    ollama pull llama3.1:8b || echo "Couldn't pull the model now; drift falls back to a word bank until one is installed."
  fi
else
  echo "Ollama isn't on the PATH; prompt drift will use the simple word-bank fallback."
fi

chmod +x start.sh
echo
echo "Setup done. Start it with ./start.sh"
echo "The image model (several GB) downloads the first time it generates."
