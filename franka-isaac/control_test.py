#!/usr/bin/env python3
import rclpy, sys
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from franka_msgs.action import GripperCommand
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration

ARM_NS = 'rightarm'          # change to 'leftarm' for the other robot
JNT_CTRL = 'joint_trajectory_controller'
GRIP_NODE = 'fr3_gripper'
ARM_JOINTS = [f'fr3_joint{i}' for i in range(1, 8)]  # 7 joints

class Teleop(Node):
    def __init__(self):
        super().__init__('teleop_franka')
        self.curr = {j: 0.0 for j in ARM_JOINTS}   # populated by callback
        self.grip = 0.0                            # current gripper width
        self.sub = self.create_subscription(
            JointState, '/joint_states', self._cb, 10)
        self.arm_cli = ActionClient(
            self, FollowJointTrajectory,
            f'/{ARM_NS}/{JNT_CTRL}/follow_joint_trajectory')
        self.grip_cli = ActionClient(
            self, GripperCommand,
            f'/{ARM_NS}/{GRIP_NODE}/gripper_action')

    def _cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in self.curr:
                self.curr[name] = pos
            elif name.endswith('_finger'):
                self.grip = pos  # use finger pos as width proxy

    def spin(self):
        self.arm_cli.wait_for_server()
        self.grip_cli.wait_for_server()
        rate = self.create_rate(10)

        while rclpy.ok():
            # 1) print current state
            q = [self.curr[j] for j in ARM_JOINTS]
            print(f'Current joints: {["%.4f" % v for v in q]}  grip: {self.grip:.4f}')
            # 2) read input
            s = input('Δq1 … Δq7 Δgrip (blank to quit)> ').strip()
            if not s:
                break
            try:
                deltas = [float(x) for x in s.split()]
                if len(deltas) != 8:
                    raise ValueError
            except ValueError:
                print('✗  Need exactly 8 numeric values');  continue
            # 3) add deltas
            q_new   = [a + d for a, d in zip(q, deltas[:7])]
            grip_new = self.grip + deltas[7]
            # send arm goal
            traj = JointTrajectory(joint_names=ARM_JOINTS,
                                   points=[JointTrajectoryPoint(
                                       positions=q_new,
                                       time_from_start=Duration(sec=0, nanosec=20000000))])  # 20 ms
            self.arm_cli.send_goal_async(
                FollowJointTrajectory.Goal(trajectory=traj))
            # send gripper goal
            self.grip_cli.send_goal_async(
                GripperCommand.Goal(width=grip_new,
                                    speed=0.1, force=20.0))
            rate.sleep()

def main():
    rclpy.init()
    try:
        Teleop().spin()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
