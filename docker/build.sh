#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
docker build --file "${REPO_ROOT}/docker/Dockerfile" \
  --tag "${IMAGE_NAME:-viwgs:cuda11.8-noetic}" \
  --build-arg USER_UID="$(id -u)" --build-arg USER_GID="$(id -g)" \
  "$@" "${REPO_ROOT}"
