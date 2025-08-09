#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Config (override by args)
# -------------------------
NS="${1:-fr3}"
ROBOT_IP="${2:-172.16.0.3}"
CTRL_YAML="${3:-/tmp/franka_ros2_ws/src/franka_ros2/franka_bringup/config/controllers.yaml}"

# Paths
WS_ROOT="${WS_ROOT:-/tmp/franka_ros2_ws}"
LOG_DIR="${LOG_DIR:-${WS_ROOT}/logs}"
mkdir -p "${LOG_DIR}"

echo "=== Prepare Franka (NS=${NS}, IP=${ROBOT_IP}) ==="
echo "YAML: ${CTRL_YAML}"
echo "Logs: ${LOG_DIR}"

# -------------------------
# Source ROS & workspace
# -------------------------
if [ -f "/opt/ros/humble/setup.bash" ]; then
  source /opt/ros/humble/setup.bash
fi
if [ -f "${WS_ROOT}/install/setup.bash" ]; then
  source "${WS_ROOT}/install/setup.bash"
fi

# Helpful: make CLI a bit quieter to avoid surprises
export RCUTILS_COLORIZED_OUTPUT=1

# -------------------------
# Helpers
# -------------------------
wait_for_service() {
  local srv="$1" ; local timeout="${2:-60}"
  echo -n "[LOG] Waiting for service ${srv} ..."
  for ((i=0; i<timeout; i++)); do
    if ros2 service list | grep -q -- "${srv}"; then
      echo " OK"
      return 0
    fi
    sleep 1
    echo -n "."
  done
  echo
  echo "[ERROR] Timeout waiting for ${srv}"
  return 1
}

wait_for_action() {
  local action="$1" ; local timeout="${2:-60}"
  echo -n "[LOG] Waiting for action ${action} ..."
  for ((i=0; i<timeout; i++)); do
    if ros2 action list | grep -q -- "${action}"; then
      echo " OK"
      return 0
    fi
    sleep 1
    echo -n "."
  done
  echo
  echo "[ERROR] Timeout waiting for ${action}"
  return 1
}

wait_for_topic() {
  local topic="$1" ; local timeout="${2:-60}"
  echo -n "[LOG] Waiting for topic ${topic} ..."
  for ((i=0; i<timeout; i++)); do
    if ros2 topic list | grep -q -- "${topic}"; then
      echo " OK"
      return 0
    fi
    sleep 1
    echo -n "."
  done
  echo
  echo "[ERROR] Timeout waiting for ${topic}"
  return 1
}

# -------------------------
# Launch bringup (background)
# -------------------------
echo "[LOG] Launching franka_bringup ..."
ros2 launch franka_bringup franka.launch.py \
  robot_ip:="${ROBOT_IP}" \
  namespace:="${NS}" \
  load_gripper:=true \
  controllers_file:="${CTRL_YAML}" \
  > "${LOG_DIR}/bringup.log" 2>&1 &

BRINGUP_PID=$!
echo "${BRINGUP_PID}" > "${LOG_DIR}/bringup.pid"
echo "   bringup PID=${BRINGUP_PID} (logs: ${LOG_DIR}/bringup.log)"

# Wait for controller_manager to be ready
wait_for_service "/${NS}/controller_manager/list_controllers" 120

# (optional) Wait for joint states to flow
wait_for_topic "/${NS}/joint_states" 30 || true

# -------------------------
# Load & configure move_to_goal
# -------------------------
CM="/${NS}/controller_manager"

echo "[LOG] Checking controllers..."
if ! ros2 control list_controllers -c "${CM}" | grep -q "^move_to_goal"; then
  echo "[LOG] Loading move_to_goal ..."
  ros2 control load_controller move_to_goal -c "${CM}"
else
  echo "[LOG] move_to_goal already loaded."
fi

# Ensure configured (inactive)
STATE_LINE="$(ros2 control list_controllers -c "${CM}" | grep '^move_to_goal' || true)"
if echo "${STATE_LINE}" | grep -q "unconfigured"; then
  echo "[LOG] Configuring move_to_goal (to inactive) ..."
  ros2 control set_controller_state move_to_goal inactive -c "${CM}"
else
  echo "[LOG] move_to_goal state: ${STATE_LINE}"
fi

# -------------------------
# Gripper: ensure node & homing
# -------------------------
if ! ros2 action list | grep -q "/${NS}/franka_gripper/move"; then
  echo "[LOG] Launching franka_gripper ..."
  ros2 launch franka_gripper gripper.launch.py \
    robot_ip:="${ROBOT_IP}" \
    namespace:="${NS}" \
    > "${LOG_DIR}/gripper.log" 2>&1 &
  GRIPPER_PID=$!
  echo "${GRIPPER_PID}" > "${LOG_DIR}/gripper.pid"
  echo "   gripper PID=${GRIPPER_PID} (logs: ${LOG_DIR}/gripper.log)"
fi

wait_for_action "/${NS}/franka_gripper/homing" 90
echo "[LOG] Gripper homing..."
ros2 action send_goal "/${NS}/franka_gripper/homing" \
  franka_msgs/action/Homing "{}" > "${LOG_DIR}/gripper_homing.log" 2>&1 || true
echo "   (homing request sent)"

# Final status
echo
echo "[LOG] Prep done."
echo "Controllers:"
ros2 control list_controllers -c "${CM}" || true
echo
echo "Tips:"
echo " - To set a goal:   ros2 service call /${NS}/move_to_goal/set_parameters rcl_interfaces/srv/SetParameters \"{parameters: [{name: 'q_goal_', value: {type: 8, double_array_value: [..7 vals..]}}, {name: 'speed_scale', value: {type: 3, double_value: 0.15}}]}\""
echo " - To activate:     ros2 control set_controller_state move_to_goal active -c ${CM}"
echo " - PIDs saved at:   ${LOG_DIR}/*.pid"
