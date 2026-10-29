#!/usr/bin/env bash
# Check status and stream logs when you reconnect your client machine
# Usage: ./check_job.sh <JOB_ID> [HOST_IP]

JOB_ID=$1
HOST_IP=${2:-"127.0.0.1"}
DASHBOARD_URL="http://${HOST_IP}:8265"
RAY_BIN=$(which ray 2>/dev/null || echo ".venv/bin/ray")

if [ -z "$JOB_ID" ]; then
    echo "Usage: ./check_job.sh <JOB_ID> [HOST_IP]"
    echo ""
    echo "Currently registered jobs on cluster (${DASHBOARD_URL}):"
    "$RAY_BIN" job list --address="${DASHBOARD_URL}"
    exit 0
fi

echo "=========================================================="
echo " Checking Status for Job: ${JOB_ID}"
echo " Host: ${DASHBOARD_URL}"
echo "=========================================================="
"$RAY_BIN" job status --address="${DASHBOARD_URL}" "${JOB_ID}"

echo ""
echo "=== Following Job Logs (Press Ctrl+C to detach without stopping job) ==="
"$RAY_BIN" job logs --address="${DASHBOARD_URL}" --follow "${JOB_ID}"
