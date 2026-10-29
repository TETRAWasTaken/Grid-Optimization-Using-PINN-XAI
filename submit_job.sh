#!/usr/bin/env bash
# Submit training job to remote Ray GPU host
# Usage: ./submit_job.sh <HOST_IP>

HOST_IP=${1:-"127.0.0.1"}
DASHBOARD_URL="http://${HOST_IP}:8265"
RAY_BIN=$(which ray 2>/dev/null || echo ".venv/bin/ray")

echo "=========================================================="
echo " Submitting Grid OPF Job to Ray Cluster at: ${DASHBOARD_URL}"
echo "=========================================================="

"$RAY_BIN" job submit \
    --address="${DASHBOARD_URL}" \
    --working-dir="." \
    -- python main.py
