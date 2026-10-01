#!/usr/bin/env bash
# Launch the Indoor VL-QKD BB84 Simulator
set -e
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt --quiet
python3 -m utils.selftest
exec python3 -m streamlit run app.py "$@"
