#!/usr/bin/env bash
# Persistent session wrapper: retain failures and package evidence on every exit.
set -euo pipefail
export PATH="/opt/conda/bin:/usr/local/cuda/bin:$PATH"
cd /workspace/tco-research
mkdir -p environment reports runs/logs
exec 9>/workspace/tco-session.lock
flock -n 9 || { echo 'An experiment session is already running'; exit 1; }
finish() {
    result=$?
    trap - EXIT
    printf '%s\n' "$result" > environment/cloud_session_exit_code.txt
    python scripts/summarize_runs.py > reports/run_ledger_generation.log 2>&1 || true
    tar -czf /workspace/tco_midterm_results.tar.gz runs reports environment
    sha256sum /workspace/tco_midterm_results.tar.gz > /workspace/tco_midterm_results.sha256
    echo "Session finished with exit code $result. Evidence is packaged; stop the pod after verifying download."
    exit "$result"
}
trap finish EXIT
bash scripts/cloud_bootstrap.sh
