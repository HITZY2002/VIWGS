#!/usr/bin/env bash
set -e
source /opt/ros/noetic/setup.bash
source /opt/viwgs_ros/setup.bash
if [[ -f /workspace/viwgs/devel/setup.bash ]]; then
  source /workspace/viwgs/devel/setup.bash
fi
exec "$@"
