#!/usr/bin/env bash
# Record teleoperation demos with cameras into a LeRobotDataset (run on the Mac, lerobot env active).
#
#   scripts/record_dataset.sh pick_place "Put the toy in the bowl" 50
#   scripts/record_dataset.sh pick_place "Put the toy in the bowl" 10 --resume=true   # add 10 more
#
# Goal-conditioned color task (one session per color, appended to the same dataset):
#   for color in red green blue; do
#     scripts/record_dataset.sh color_blocks "Put the ${color} block in the bowl" 20 --resume=true
#   done
#
# Joints are recorded in degrees (use_degrees=true) so the dataset, the policy client and the
# Isaac Sim twin all share one unit. Camera indices: run `lerobot-find-cameras opencv` once.
set -euo pipefail

NAME="${1:?dataset name, e.g. pick_place}"
TASK="${2:?task sentence}"
EPISODES="${3:-50}"
shift $(( $# < 3 ? $# : 3 ))

HF_USER="${HF_USER:-kayarecepomer}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/tty.usbmodem5B610352021}"
LEADER_PORT="${LEADER_PORT:-/dev/tty.usbmodem5B610373041}"
WRIST_CAM="${WRIST_CAM:-0}"
FRONT_CAM="${FRONT_CAM:-1}"

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id=kayas_follower \
  --robot.use_degrees=true \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: ${FRONT_CAM}, width: 640, height: 480, fps: 30} }" \
  --teleop.type=so101_leader \
  --teleop.port="${LEADER_PORT}" \
  --teleop.id=kayas_leader \
  --teleop.use_degrees=true \
  --display_data=true \
  --dataset.repo_id="${HF_USER}/so101_${NAME}" \
  --dataset.single_task="${TASK}" \
  --dataset.num_episodes="${EPISODES}" \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=15 \
  --dataset.push_to_hub=true \
  "$@"
