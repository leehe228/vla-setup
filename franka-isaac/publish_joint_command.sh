#!/bin/bash

JOINTS="['panda_joint1','panda_joint2','panda_joint3','panda_joint4','panda_joint5','panda_joint6','panda_joint7']"

while true; do
  NOW_SEC=$(date +%s)
  NOW_NSEC=$(($(date +%N) % 1000000000))

  POSITIONS=$(python3 -c "
import random
print([round(random.uniform(-2.0, 2.0), 3) for _ in range(7)])
")

  ros2 topic pub /joint_command sensor_msgs/msg/JointState "{
  header: {
    stamp: { sec: $NOW_SEC, nanosec: $NOW_NSEC },
    frame_id: ''
  },
  name: $JOINTS,
  position: $POSITIONS,
  velocity: [],
  effort: []
}" -1

  sleep 1
done
