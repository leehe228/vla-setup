#!/usr/bin/env python3
"""
TinyVLA + ROS 2 Humble + Isaac Sim 4.5
CPU-only OK (TinyVLA-S/B/H 선택 가능)
"""
import os, time, threading, argparse, numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import cv2

# TinyVLA modules (import 경로는 PYTHONPATH로 해결)
from llava_pythia.conversation import conv_templates
from llava_pythia.model.builder import load_pretrained_model          # 백본 로드
from tinyvla.policy_heads.act_diffusion import ActPolicy             # diffusion head

# ---------- 설치 경로 자동 검색 (checkout 폴더 기준) -------------
ROOT = os.getenv("TINYVLA_HOME", os.path.dirname(os.path.abspath(__file__)))
BACKBONE = os.path.join(ROOT, "checkpoints/llava_pythia_400m")  # 사전 제공된 weight 경로
POLICY   = os.path.join(ROOT, "checkpoints/diffusion_head")     # TinyVLA diffusion head

# ------------- 1 Hz 컨트롤 노드 ---------------------------------
class TinyVLAROS(Node):
    def __init__(self, max_iter=100):
        super().__init__("tinyvla_controller")
        self.bridge = CvBridge()
        self.image  = None
        self.q_prev = np.zeros(7)
        self.max_iter = max_iter
        self.pub = self.create_publisher(JointState, "/joint_command", 10)
        self.create_subscription(Image, "/rgb", self.img_cb, 10)

        # TinyVLA 로드 (CPU) -----------------------------------
        templ = conv_templates["pythia"].copy()
        self.tokenizer, self.backbone, self.image_proc, _ = \
            load_pretrained_model(BACKBONE, None, "pythia", fp16=False, device="cpu")
        self.policy = ActPolicy.load_from_pretrained(POLICY, device="cpu")
        self.conv_template = templ

        self.get_logger().info("TinyVLA ready – enter instruction in terminal...")
        threading.Thread(target=self.prompt_loop, daemon=True).start()

    # 이미지 콜백 → 최신 프레임 저장
    def img_cb(self, msg: Image):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge err {e}")

    # 사용자 프롬프트 루프
    def prompt_loop(self):
        it = 0
        while rclpy.ok() and it < self.max_iter:
            prompt = input("\n▶ Instruction: ").strip()
            if not prompt: continue
            if self.image is None:
                print("  (waiting camera...)"); time.sleep(0.2); continue

            # 1) 이미지 전처리
            img_tensor = self.image_proc.preprocess(self.image,
                                                    return_tensors='pt')['pixel_values']
            # 2) 프롬프트 템플릿 조립
            conv = self.conv_template.copy()
            conv.append_message(conv.roles[0], "<image>\n"+prompt)
            conv.append_message(conv.roles[1], None)
            input_ids = self.tokenizer(conv.get_prompt(), return_tensors='pt').input_ids

            # 3) backbone → diffusion head
            with torch.no_grad():
                feat = self.backbone(images=img_tensor, input_ids=input_ids)
                cart6d = self.policy(feat).cpu().numpy()[0]    # [dx..drz, grip]

            # 4) 6-DoF → 7 Franka 엉성 IK (직접 델타)
            dq = np.zeros(7)
            dq[:3] = cart6d[:3] * 0.2                # 위치 비율
            dq[3:6] = cart6d[3:6] * 0.1              # 각도 비율
            self.q_prev = np.clip(self.q_prev + dq,
                                  [-2.8, -1.76, -2.8, -3.0, -2.9, -3.0, -2.9],
                                  [ 2.8,  1.76,  2.8,  3.0,  2.9,  3.0,  2.9])

            # 5) Publish
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = [f"panda_joint{i}" for i in range(1,8)]
            js.position = self.q_prev.tolist()
            self.pub.publish(js)
            self.get_logger().info(f"{it:03d}  dq={dq.round(3)}")
            it += 1
            time.sleep(1.0)

# ---------------- main -----------------
if __name__ == "__main__":
    rclpy.init()
    TinyVLAROS(max_iter=100)
    rclpy.spin()
    rclpy.shutdown()
