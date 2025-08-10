#!/usr/bin/env python3
"""
moveit_data_recorder.py
- Record exterior/wrist images, joint(7), gripper width, optional EE pose, and their deltas every 0.1s
- Meant to run while you teleoperate the robot via MoveIt (Servo/RViz etc.)
"""

import os, sys, time, json, io, base64, threading
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.time import Time as RclTime
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

# ========= User constants =========
NAMESPACE        = "fr3"
SESSION_ROOT     = "./dataset_sessions"    # 세션들이 누적될 디렉터리
STEP_SEC         = 0.10                    # 0.1s(10Hz)
MAX_DURATION_SEC = None                    # None=무한, 아니면 N초 후 종료
PROMPT           = "Pick up the box."

# Cameras
CAM_FRONT_INDEX  = 0
CAM_WRIST_INDEX  = 10
IMG_SIZE         = 244
CAM_WARMUP_SEC   = 0.15
CAM_FLUSH_MS     = 120

# TF (선택): EE pose 기록하고 싶을 때 링크명
USE_TF_POSE      = True
BASE_LINK        = f"{NAMESPACE}_link0"
EEF_LINK         = f"{NAMESPACE}_hand_tcp"   # 설정에 따라 fr3_link8, fr3_hand_tcp 등으로 조정하세요.

# ======== Camera helpers =========
def _looks_green(frame: np.ndarray) -> bool:
    b, g, r = cv2.split(frame)
    gm, rm, bm = float(g.mean()), float(r.mean()), float(b.mean())
    return (gm > 60.0) and (gm > 1.3 * rm) and (gm > 1.3 * bm)

def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    # warm-up
    t_end = time.time() + CAM_WARMUP_SEC
    while time.time() < t_end:
        cap.read()
        time.sleep(0.01)
    return cap

def grab_244(cap: cv2.VideoCapture) -> np.ndarray:
    # flush buffer (avoid stale frame)
    t_end = time.time() + (CAM_FLUSH_MS / 1000.0)
    while time.time() < t_end:
        cap.grab()
    ok = cap.grab()
    if not ok:
        raise RuntimeError("Failed to grab")
    ok, frame = cap.retrieve()
    if not ok:
        raise RuntimeError("Failed to retrieve frame")
    # if green-ish frame appears, retry a few times
    tries = 5
    while _looks_green(frame) and tries > 0:
        time.sleep(0.03)
        cap.grab()
        ok, f2 = cap.retrieve()
        if ok:
            frame = f2
        tries -= 1
    # center-crop square then resize to 244x244; return RGB
    h, w = frame.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    crop = frame[y0:y0+s, x0:x0+s]
    resized_bgr = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized_bgr[:, :, ::-1]

# ======== ROS node =========
class Recorder(Node):
    def __init__(self, session_dir: str):
        super().__init__("moveit_data_recorder")
        self.arm_joint_names: List[str] = [f"{NAMESPACE}_joint{i}" for i in range(1, 8)]
        self._arm_js = None       # np.ndarray (7,)
        self._grip_js = None      # np.ndarray (len 1 or 2)
        self._lock = threading.Lock()

        self.create_subscription(JointState, f"/{NAMESPACE}/joint_states", self._on_arm_js, 50)
        self.create_subscription(JointState, f"/{NAMESPACE}/franka_gripper/joint_states", self._on_grip_js, 10)

        # TF (optional)
        self.tf_buf = Buffer()
        self.tf_lst = TransformListener(self.tf_buf, self, spin_thread=True)

        self.session_dir = session_dir
        os.makedirs(self.session_dir, exist_ok=True)
        self.meta_path = os.path.join(self.session_dir, "meta.jsonl")
        self.counter = 0
        self.prev_q: Optional[np.ndarray] = None
        self.prev_grip: Optional[float] = None

        # Open cameras
        self.cap_front = open_camera(CAM_FRONT_INDEX)
        self.cap_wrist = open_camera(CAM_WRIST_INDEX)
        print(f"[INFO] Cameras opened: front={CAM_FRONT_INDEX}, wrist={CAM_WRIST_INDEX}")
        print(f"[INFO] Saving to: {self.session_dir}")
        print(f"[INFO] Prompt: {PROMPT}")

    # --- joint callbacks
    def _on_arm_js(self, msg: JointState):
        idx, name_to_idx, ok = [], {n: i for i, n in enumerate(msg.name)}, True
        for n in self.arm_joint_names:
            if n not in name_to_idx:
                ok = False; break
            idx.append(name_to_idx[n])
        if not ok:
            return
        vals = np.array([msg.position[i] for i in idx], dtype=np.float64)
        with self._lock:
            self._arm_js = vals

    def _on_grip_js(self, msg: JointState):
        if not msg.position:
            return
        with self._lock:
            self._grip_js = np.array(msg.position, dtype=np.float64)

    # --- state read
    def get_state(self, wait_sec=1.0) -> Tuple[np.ndarray, float]:
        t0 = time.time()
        while time.time() - t0 < wait_sec:
            with self._lock:
                arm = None if self._arm_js is None else self._arm_js.copy()
                gj  = None if self._grip_js is None else self._grip_js.copy()
            if arm is not None and gj is not None:
                if len(gj) >= 2:
                    width = float(np.clip(gj[0] + gj[1], 0.0, 0.08))
                else:
                    width = float(np.clip(2.0 * gj[0], 0.0, 0.08))
                return arm, width
            rclpy.spin_once(self, timeout_sec=0.01)
        raise RuntimeError("Timed out waiting for joint/gripper states.")

    # --- TF pose
    def lookup_ee_pose(self) -> Optional[Tuple[List[float], List[float]]]:
        if not USE_TF_POSE:
            return None
        try:
            ts: TransformStamped = self.tf_buf.lookup_transform(
                BASE_LINK, EEF_LINK, rclpy.time.Time())
            t = ts.transform.translation
            q = ts.transform.rotation
            pos = [t.x, t.y, t.z]
            quat = [q.x, q.y, q.z, q.w]
            return pos, quat
        except Exception:
            return None

    # --- capture one sample
    def capture_once(self) -> bool:
        t_stamp = time.time()
        q, grip = self.get_state(wait_sec=1.0)
        ext = grab_244(self.cap_front)
        wrist = grab_244(self.cap_wrist)

        # filenames
        tag = f"{self.counter:06d}"
        ext_path = os.path.join(self.session_dir, f"exterior_{tag}.png")
        wrist_path = os.path.join(self.session_dir, f"wrist_{tag}.png")
        # save images (RGB → BGR)
        cv2.imwrite(ext_path, cv2.cvtColor(ext, cv2.COLOR_RGB2BGR))
        cv2.imwrite(wrist_path, cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))

        # deltas
        if self.prev_q is None:
            dq = np.zeros_like(q)
            dgrip = 0.0
        else:
            dq = q - self.prev_q
            dgrip = grip - (self.prev_grip if self.prev_grip is not None else grip)

        self.prev_q = q.copy()
        self.prev_grip = float(grip)

        # optional EE pose
        ee_pose = self.lookup_ee_pose()  # (pos, quat) or None

        rec = {
            "t": t_stamp,
            "idx": self.counter,
            "prompt": PROMPT,
            "q": q.tolist(),             # 7
            "grip": float(grip),         # width [m]
            "dq": dq.tolist(),           # Δq (next - prev)
            "dgrip": float(dgrip),       # Δgrip
            "ext_image": os.path.basename(ext_path),
            "wrist_image": os.path.basename(wrist_path),
        }
        if ee_pose is not None:
            rec["ee_position"] = ee_pose[0]  # [x,y,z]
            rec["ee_quaternion"] = ee_pose[1]# [qx,qy,qz,qw]

        with open(self.meta_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        print(f"[{self.counter:06d}] q={np.array2string(q, precision=4)} "
              f"grip={grip:.4f}  dq={np.array2string(dq, precision=4)} dgrip={dgrip:.4f}")
        self.counter += 1
        return True

    def close(self):
        try:
            self.cap_front.release()
            self.cap_wrist.release()
        except Exception:
            pass

def main():
    rclpy.init()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir = os.path.join(SESSION_ROOT, f"session_{ts}")
    node = Recorder(session_dir)

    t0 = time.time()
    try:
        while True:
            tick = time.perf_counter()
            node.capture_once()
            # pacing to ~STEP_SEC
            elapsed = time.perf_counter() - tick
            sleep_s = max(0.0, STEP_SEC - elapsed)
            time.sleep(sleep_s)
            if MAX_DURATION_SEC is not None and (time.time() - t0) > MAX_DURATION_SEC:
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        node.close()
        rclpy.shutdown()
        print(f"[DONE] Saved dataset to: {node.session_dir}")

if __name__ == "__main__":
    main()
