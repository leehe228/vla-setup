#!/usr/bin/env python3
"""
Teleoperation node for a single Franka Research 3 arm.

Type eight floating-point deltas: 7 joint + 1 gripper width.
They are added to the current state and streamed as a 20 ms
FollowJointTrajectory plus GripperCommand goal.

Empty line or invalid input terminates the loop.
"""

import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory        #  [oai_citation:6‡ROS Documentation](https://docs.ros.org/en/noetic/api/control_msgs/html/action/FollowJointTrajectory.html?utm_source=chatgpt.com)
from franka_msgs.action import GripperCommand                #  [oai_citation:7‡ROS Documentation](https://docs.ros.org/en/humble/p/franka_msgs/?utm_source=chatgpt.com)
from builtin_interfaces.msg import Duration

ARM_NS_DEFAULT = 'rightarm'                                  # change via CLI remap
JTC_NAME = 'joint_trajectory_controller'
GRIP_NODE = 'fr3_gripper'
ARM_JOINTS = [f'fr3_joint{i}' for i in range(1, 8)]           # seven joints

class Teleop(Node):
    def __init__(self):
        super().__init__('teleop_franka')
        self.declare_parameter('arm_ns', ARM_NS_DEFAULT)
        self.ns = self.get_parameter('arm_ns').value

        self.curr = {j: 0.0 for j in ARM_JOINTS}
        self.grip = 0.0

        # ① state subscription
        self.create_subscription(JointState, '/joint_states', self._cb_state, 10)  #  [oai_citation:8‡GitHub](https://github.com/frankaemika/franka_ros/issues/303?utm_source=chatgpt.com)

        # ② action clients
        arm_action = f'/{self.ns}/{JTC_NAME}/follow_joint_trajectory'
        grip_action = f'/{self.ns}/{GRIP_NODE}/gripper_action'
        self.arm_cli = ActionClient(self, FollowJointTrajectory, arm_action)
        self.grip_cli = ActionClient(self, GripperCommand, grip_action)

        self.get_logger().info(f'Waiting for action servers under namespace "{self.ns}" …')
        self.arm_cli.wait_for_server()
        self.grip_cli.wait_for_server()
        self.get_logger().info('✓ Servers ready')

    # --- callbacks ---------------------------------------------------------
    def _cb_state(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in self.curr:
                self.curr[name] = pos
            elif name.endswith('_finger'):          # use finger joint as width proxy
                self.grip = pos

    # --- main loop ---------------------------------------------------------
    def run(self):
        rate = self.create_rate(10)                # 10 Hz print cadence
        while rclpy.ok():
            q = [self.curr[j] for j in ARM_JOINTS]
            print(f'Current q: {["%.4f" % v for v in q]} | grip: {self.grip:.4f}')
            s = input('Δq1 … Δq7 Δgrip > ').strip()
            if not s:
                break
            try:
                d = [float(x) for x in s.split()]
                if len(d) != 8:
                    raise ValueError
            except ValueError:
                print('✗ Need exactly 8 numeric values – quitting')
                break

            q_new = [a + dv for a, dv in zip(q, d[:7])]
            grip_new = self.grip + d[7]

            # ③ publish trajectory (20 ms horizon)
            traj = JointTrajectory(
                joint_names=ARM_JOINTS,
                points=[JointTrajectoryPoint(
                    positions=q_new,
                    time_from_start=Duration(sec=0, nanosec=20_000_000))]
            )                                       # joint_trajectory_controller docs  [oai_citation:9‡control.ros.org](https://control.ros.org/foxy/doc/ros2_controllers/joint_trajectory_controller/doc/userdoc.html?utm_source=chatgpt.com)
            self.arm_cli.send_goal_async(
                FollowJointTrajectory.Goal(trajectory=traj))

            # ④ publish gripper goal
            self.grip_cli.send_goal_async(
                GripperCommand.Goal(width=grip_new, speed=0.1, force=20.0))  # width/force usage  [oai_citation:10‡GitHub](https://github.com/frankaemika/franka_ros/issues/303?utm_source=chatgpt.com)
            rate.sleep()

# -------------------------------------------------------------------------
def main(argv=None):
    rclpy.init(args=argv)
    Teleop().run()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
