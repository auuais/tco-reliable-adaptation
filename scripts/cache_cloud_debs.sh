#!/usr/bin/env bash
# Execute in a fresh pinned base image with /var/cache/apt/archives mounted locally.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --download-only --no-install-recommends build-essential ninja-build git libgl1 libglib2.0-0 openssh-server
