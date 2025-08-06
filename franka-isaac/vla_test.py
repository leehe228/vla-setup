#!/usr/bin/env python3
"""
SmolVLA ⇆ ROS 2  control loop
  ① 이미지 /rgb subscribe
  ② 사용자 프롬프트 입력
  ③ SmolVLA로 6-DoF 델타 예측
  ④ 간단 IK → 7-joint 명령 → /joint_command publish
반복 100회 후 종료
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import cv2
import numpy as np
from datetime import datetime
import torch
from smolvla import SmolVLA, Processor   # pip install smolvla
from threading import Thread, Event

JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3",
    "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
]

class VLASubscriber(Node):
    """Keeps latest camera frame & joint state; publishes joint commands."""
    def __init__(self):
        super().__init__("vla_subscriber")
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image, "/rgb", self._img_cb, 10)
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_cb, 10)
        self.cmd_pub = self.create_publisher(
            JointState, "/joint_command", 10)

        self.latest_img = None
        self.latest_jpos = np.zeros(7)
        self.img_ready = Event()

    # === Callbacks ===
    def _img_cb(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_img = cv_img
            self.img_ready.set()
        except Exception as e:
            self.get_logger().error(f"bridge error: {e}")

    def _joint_cb(self, msg: JointState):
        # map by name to fixed order
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        self.latest_jpos = np.array([msg.position[name_to_idx[n]] for n in JOINT_NAMES])

    # === Publish helper ===
    def send_joint_cmd(self, pos: np.ndarray):
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = JOINT_NAMES
        out.position = pos.tolist()
        self.cmd_pub.publish(out)

# ---------- Simple differential IK (Jacobian ≈ identity) ----------
def ee_delta_to_joints(delta_6: np.ndarray, q_curr: np.ndarray, eta: float = 0.2):
    """
    Very crude mapper: first 6 joint increments proportional to dx..drz,
    keep joint7 unchanged.
    """
    scale_xyz = 0.4     # rad per metre (tuned)
    scale_rpy = 0.5     # rad per rad
    dq = np.zeros(7)
    dq[:3] = scale_xyz * delta_6[:3]
    dq[3:6] = scale_rpy * delta_6[3:]
    q_new = q_curr + eta * dq
    # clamp within Franka limits (-2.8 ~ 2.8 rad for simplicity)
    return np.clip(q_new, -2.8, 2.8)

# ---------- Spin rclpy in a background thread ----------
def spin_thread(node: Node):
    rclpy.spin(node)

# ---------- Main ----------
def main():
    rclpy.init()
    node = VLASubscriber()

    # start ROS spinning
    t = Thread(target=spin_thread, args=(node,), daemon=True)
    t.start()

    # load SmolVLA (CPU)
    print("Loading SmolVLA (CPU)…")
    device = torch.device("cpu")
    model = SmolVLA.from_pretrained("lerobot/smolvla_base").to(device)
    processor = Processor.from_pretrained("lerobot/smolvla_base")
    print("Loaded.")

    # wait for first image
    print("Waiting for /rgb frames…")
    node.img_ready.wait()

    for step in range(1, 101):
        prompt = input(f"[{step}/100]  Enter instruction > ").strip()
        if prompt == "":
            print("  (empty input → skipping)")
            continue

        # grab latest img snapshot
        bgr = node.latest_img.copy()
        # resize to 224×224 (SmolVLA default)
        rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)

        # model inference
        inputs = processor(images=rgb, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # SmolVLA returns (B,6) tensor of deltas (already scaled to metres/rad)
        delta = outputs.action[0].cpu().numpy()   # (dx,dy,dz,drx,dry,drz)
        print("  model Δ:", np.round(delta, 3))

        # map to 7-joint command
        q_des = ee_delta_to_joints(delta, node.latest_jpos)
        node.send_joint_cmd(q_des)
        print("  published joint command\n")

    print("Stopping…")
    rclpy.shutdown()
    t.join()

if __name__ == "__main__":
    main()