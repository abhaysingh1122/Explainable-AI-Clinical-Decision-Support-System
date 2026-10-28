#!/usr/bin/env bash
# Run with: ./run.sh   (first run sets up Python packages)
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
    echo "Setting up for the first time..."
    python3 -m venv .venv
    . .venv/bin/activate
    pip install -r requirements.txt
else
    . .venv/bin/activate
fi
python run.py "$@"
