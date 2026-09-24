#!/usr/bin/env bash
# Start HandsToVoice from anywhere: ./run.sh [--camera N]
cd "$(dirname "$(readlink -f "$0")")" || exit 1
if [ ! -x venv/bin/python3 ]; then
    echo "HandsToVoice: venv/ not found. Set it up first:" >&2
    echo "  python3 -m venv venv && venv/bin/python3 -m pip install -r requirements.txt" >&2
    exit 1
fi
exec venv/bin/python3 main.py "$@"
