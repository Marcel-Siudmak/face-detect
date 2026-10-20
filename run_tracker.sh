#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo "    Minecraft HeadTracker 6-DoF & Off-Axis Portal"
echo "========================================================"
echo "[+] Starting Face/Head Tracking system..."
echo "[+] Target UDP streaming: 127.0.0.1:4242"
echo "[+] Controls: Press 'q' in tracking window to exit, 'c' to calibrate neutral Y."
echo "========================================================"

if [ -f ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_EXEC="python3"
else
    echo "[-] Error: Python executable not found!"
    exit 1
fi

exec "$PYTHON_EXEC" detection.py "$@"
