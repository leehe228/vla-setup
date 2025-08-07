import cv2, base64, requests, numpy as np
from PIL import Image
import io, json

SERVER_URL = "http://221.139.43.89:8000/api/infer"

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ----- stub: replace with actual subscriber callback or ROS bag read -----
def get_ros_data():
    exterior = np.zeros((244,244,3), np.uint8)   # TODO replace with real cv_image
    wrist    = np.zeros((244,244,3), np.uint8)
    joint    = [0.0]*7
    grip     = [0.0]
    prompt   = "Pick up the red cube and move it to the bin."
    return exterior, wrist, joint, grip, prompt

def main():
    ext_np, wrist_np, joint, grip, prompt = get_ros_data()
    ext_b64   = pil_to_b64(Image.fromarray(ext_np))
    wrist_b64 = pil_to_b64(Image.fromarray(wrist_np))

    payload = {
        "exterior_image_1_left": ext_b64,
        "wrist_image_left":      wrist_b64,
        "joint_position":        joint,
        "gripper_position":      grip,
        "prompt":                prompt,
    }
    resp = requests.post(SERVER_URL, json=payload, timeout=10)
    resp.raise_for_status()
    actions = resp.json()["actions"]
    print("Received action token:", actions)
    # ⚙️  here: convert & publish JointState as needed

if __name__ == "__main__":
    main()
