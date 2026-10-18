#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "${REPO_ROOT}/output"
GUI_ARGS=()
if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
  GUI_ARGS+=(-e DISPLAY -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro)
  XAUTH_FILE=${XAUTHORITY:-${HOME}/.Xauthority}
  if [[ -f "${XAUTH_FILE}" ]]; then
    GUI_ARGS+=(-e XAUTHORITY=/tmp/viwgs.xauth -v "${XAUTH_FILE}:/tmp/viwgs.xauth:ro")
  fi
fi
# Extra arguments are Docker flags, e.g. -v /path/to/data:/data:ro.
docker run --rm -it --name "${CONTAINER_NAME:-viwgs-dev}" --gpus all --network host --shm-size=2g \
  "${GUI_ARGS[@]}" \
  -v "${REPO_ROOT}:/workspace/viwgs" \
  -v "${REPO_ROOT}/output:/home/viwgs/.ros/viwgs" \
  -w /workspace/viwgs "$@" "${IMAGE_NAME:-viwgs:cuda11.8-noetic}" bash
