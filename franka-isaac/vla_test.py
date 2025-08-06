#!/usr/bin/env python3
"""
ROS 2 Humble  + Isaac Sim 4.5  + TinyVLA (CPU).
"""
import os, time, argparse, threading, numpy as np
import rclpy, torch
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge

# ── TinyVLA 내부 모듈 직접 import ────────────────────────────
from llava_pythia.conversation import conv_templates
from llava_pythia.model.builder import load_pretrained_model
from policy_heads.act_diffusion import ActPolicy          # editable-install됨

ROOT = os.getenv("TINYVLA_HOME", os.path.dirname(__file__))
BACKBONE_CKPT = f"{ROOT}/checkpoints/llava_pythia_400m"
HEAD_CKPT     = f"{ROOT}/checkpoints/diffusion_head"

# ── 간단 6-DoF→7j 변환 ─────────────────────────────────────
LIMITS = np.array([2.8,1.76,2.8,3.0,2.9,3.0,2.9])
def cart6d_to_joints(delta, q_prev):
    scale = np.array([.2,.2,.2,.1,.1,.1,0])
    q = np.clip(q_prev + delta*scale, -LIMITS, LIMITS)
    return q

class TinyVLANode(Node):
    def __init__(self, max_iter=100):
        super().__init__("tinyvla_controller")
        self.bridge, self.image = CvBridge(), None
        self.q_prev, self.max_iter = np.zeros(7), max_iter
        self.pub = self.create_publisher(JointState, "/joint_command", 10)
        self.create_subscription(Image, "/rgb", self.img_cb, 10)

        # ― 모델 로드 (CPU) ―
        templ = conv_templates["pythia"].copy()
        self.tok, self.backbone, self.ipp, _ = load_pretrained_model(
            BACKBONE_CKPT, None, "pythia", fp16=False, device="cpu")
        self.head = ActPolicy.load_from_pretrained(HEAD_CKPT, device="cpu")
        self.conv_template = templ
        self.get_logger().info("TinyVLA ready. Type instruction ↵")
        threading.Thread(target=self.prompt_loop, daemon=True).start()

    def img_cb(self, msg):          # 카메라 최신 프레임 저장
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def prompt_loop(self):
        for step in range(self.max_iter):
            txt = input("Instruction> ").strip()
            if not txt or self.image is None: continue

            # 이미지 전처리
            img_t = self.ipp.preprocess(self.image, return_tensors='pt')['pixel_values']

            # LLaVA 프롬프트 생성
            conv = self.conv_template.copy()
            conv.append_message(conv.roles[0], "<image>\n"+txt)
            conv.append_message(conv.roles[1], None)
            ids = self.tok(conv.get_prompt(), return_tensors='pt').input_ids

            # 추론
            with torch.no_grad():
                feat = self.backbone(images=img_t, input_ids=ids)
                act6 = self.head(feat)[0].cpu().numpy()   # [dx…drz,grip]

            dq = cart6d_to_joints(act6, self.q_prev)
            self.q_prev = dq

            # JointState publish
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = [f"panda_joint{i}" for i in range(1,8)]
            js.position = dq.tolist()
            self.pub.publish(js)
            self.get_logger().info(f"{step}: {dq.round(3)}")
            time.sleep(1)
            
def main():
    rclpy.init(); TinyVLANode(); rclpy.spin(); rclpy.shutdown()
    
if __name__ == "__main__":
    main()
