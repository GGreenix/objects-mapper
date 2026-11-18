#!/usr/bin/env bash
# Start FRC Field Mapper (Linux / macOS). The first run sets everything up.
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "Setting up Field Mapper for the first time..."
  PY=""
  for candidate in python3.11 python3.12 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)'; then
      PY="$candidate"
      break
    fi
  done
  if [ -z "$PY" ]; then
    echo "Python 3.11-3.13 is needed. Install it (Ubuntu: sudo apt install python3.11 python3.11-venv) and run ./start.sh again."
    exit 1
  fi
  echo "Creating the Python environment with $PY ..."
  "$PY" -m venv .venv
fi

exec .venv/bin/python launch.py "$@"
