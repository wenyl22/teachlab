#!/usr/bin/env bash
# Build SE-Bench's sandbox image and start it on localhost:8111, as in SE-Bench's sandbox_build_and_run.sh.
# Needs data/sebench/zwc (bash teachlab/download_sebench.sh). Stop it with: docker rm -f sebench_sandbox
set -euo pipefail
cd "$(dirname "$0")/../.."
CTX=$(mktemp -d)
trap 'rm -rf "$CTX"' EXIT
cp -r data/sebench/zwc "$CTX/zwc"
cp teachlab/sandbox/worker.py teachlab/sandbox/Dockerfile "$CTX/"
docker build -t sebench_sandbox:latest "$CTX"
docker rm -f sebench_sandbox >/dev/null 2>&1 || true
docker run -d --name sebench_sandbox -p 8111:8111 sebench_sandbox:latest
echo "sandbox running at http://localhost:8111/run"
