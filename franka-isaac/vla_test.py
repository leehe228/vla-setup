#!/usr/bin/env python3
"""
TinyVLA ⇆ ROS2 ⇆ Isaac-Sim demo.
Tested with:  Ubuntu 22.04, ROS 2 Humble, Python 3.10 (conda env 'tinyvla')

Run:
  conda activate tinyvla
  python tinyvla_pipeline.py --model tinyvla_b --steps 100
"""
import argparse, sys, time, threading, random, math
from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import numpy as np
import cv2

# --- TinyVLA -------------------------------------------------------
from tinyvla import TinyVLA, Processor        # provided by `pip install -e .`

# ---- simple joint delta helper (very crude!) ----------------------
PANDA_JOINT_LIMIT = np.array([2.8, 1.76, 2.8, -0.05, 2.9, 3.0, 2.9])
def cartesian_to_joint(deltas_xyzr: np.ndarray,
                       q_prev: np.ndarray,
                       gain_pos=0.4,
                       gain_rot=0.2) -> np.ndarray:
    """Map 6-DoF delta to 7 Panda joint increments (heuristic)."""
    dx, dy, dz, drx, dry, drz = deltas_xyzr[:6]
    # naïve mapping: first 3 joints → xyz, next 3 → rot, joint7 = grip placeholder
    q_delta = np.array([dx, dy, dz, drx, dry, drz, 0.0])
    q_delta[:3] *= gain_pos
    q_delta[3:6] *= gain_rot
    q_next = np.clip(q_prev + q_delta, -PANDA_JOINT_LIMIT, PANDA_JOINT_LIMIT)
    return q_next

# ---- ROS-2 Node ---------------------------------------------------
class TinyVLAPipeline(Node):
    def __init__(self, model_name: str, steps: int):
        super().__init__('tinyvla_pipeline')
        self.bridge   = CvBridge()
        self.image    = None            # latest BGR image
        self.q_prev   = np.zeros(7)     # keep last joint command
        self.steps    = steps

        # subscriptions / publications
        self.create_subscription(Image, '/rgb',
                                 self.img_cb, 10)
        self.pub = self.create_publisher(JointState,
                                         '/joint_command', 10)

        # Load model (CPU by default)
        self.processor = Processor.from_pretrained(
            f"tinyvla/backbones/{model_name}")
        self.model     = TinyVLA.from_pretrained(
            f"tinyvla/weights/{model_name}")
        self.get_logger().info(f"TinyVLA '{model_name}' loaded.")

        # user prompt thread
        threading.Thread(target=self.prompt_loop,
                         daemon=True).start()

    def img_cb(self, msg: Image):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge: {e}')

    # ---------------------------------------------------------------
    def prompt_loop(self):
        count = 0
        while count < self.steps and rclpy.ok():
            text = input("\nInstruction > ").strip()
            if not text:
                print("empty prompt – skipping")
                continue
            if self.image is None:
                print("waiting for first camera frame …")
                time.sleep(0.2); continue

            # 1) preprocess
            _in = self.processor(images=[self.image],
                                 text=[text],
                                 return_tensors="pt")

            # 2) model inference  ----------------------------------
            with self.get_logger().profiling():
                out = self.model(**_in)

            action = out.action[0].detach().cpu().numpy()  # [dx…drz, grip]
            # clamp & scale
            action[:6] = np.clip(action[:6], -1, 1) * 0.05  # 5 cm / rad max
            grip = int(np.sign(action[6]))                  # -1,0,1
            # 3) convert to 7-joint
            q_cmd = cartesian_to_joint(action, self.q_prev)
            self.q_prev = q_cmd

            # 4) publish
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = [f'panda_joint{i}' for i in range(1, 8)]
            js.position = q_cmd.tolist()
            self.pub.publish(js)
            self.get_logger().info(f'[{count}] sent joints {q_cmd.round(3)}')
            count += 1
            time.sleep(1.0)  # 1 Hz

# ------------------- main ------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='tinyvla_b',
                        help='tinyvla_s | tinyvla_b | tinyvla_h etc.')
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()

    rclpy.init()
    node = TinyVLAPipeline(args.model, args.steps)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
