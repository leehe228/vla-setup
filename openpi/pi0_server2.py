import os
import base64
import io
import time
import threading
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---- openpi imports (as-is) ----
from openpi.models import model as _model                 # noqa: F401
from openpi.policies import droid_policy, policy_config  # noqa: F401
from openpi.shared import download
from openpi.training import config as _config

# ---------- display / save config ----------
SHOW_WINDOW_ENV = os.environ.get("SHOW_WINDOW", "1") == "1"
HAS_DISPLAY = bool(os.environ.get("DISPLAY"))
SHOW_WINDOW = SHOW_WINDOW_ENV and HAS_DISPLAY
WINDOW_NAME = "Pi0 Request Monitor"

IMG_EXPECTED_SIZE = (244, 244)

SAVE_DIR = os.environ.get("SAVE_DIR", "./received_images")
os.makedirs(SAVE_DIR, exist_ok=True)

# 디스플레이 공유 상태 (RGB 보관)
_disp_lock = threading.Lock()
_disp_ready_event = threading.Event()
_disp_data = {
    "ext": None,      # np.ndarray (H,W,3) RGB
    "wrist": None,    # np.ndarray (H,W,3) RGB
    "prompt": "",
    "joint": None,    # np.ndarray (7,)
    "grip": None,     # float
    "ts": 0.0,
}
_stop_display = False


def _safe_resize_rgb(img: np.ndarray, size=(244, 244)) -> np.ndarray:
    if img is None:
        return None
    if img.shape[:2] != size:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img


def _draw_overlay(bgr: np.ndarray, prompt: str, joint: Optional[np.ndarray], grip: Optional[float]):
    y = 24
    lh = 22
    cv2.putText(bgr, f"Prompt: {prompt[:90]}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    y += lh
    if joint is not None:
        cv2.putText(
            bgr,
            "Joint: " + np.array2string(joint, precision=3, max_line_width=120),
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 255, 200),
            1,
            cv2.LINE_AA,
        )
        y += lh
    if grip is not None:
        cv2.putText(
            bgr,
            f"Gripper width: {grip:.4f} m",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 255),
            1,
            cv2.LINE_AA,
        )


def _display_loop():
    if not _disp_ready_event.wait(timeout=10.0):
        print("Display thread: timeout waiting for first frame. Exit.")
        return
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1000, 520)
    except Exception as e:
        print(f"Display thread: failed to create window (headless?): {e}")
        return

    print("Display thread: started.")
    last_ts = -1.0
    while not _stop_display:
        try:
            with _disp_lock:
                ext   = _disp_data["ext"]
                wrist = _disp_data["wrist"]
                prompt= _disp_data["prompt"]
                joint = _disp_data["joint"]
                grip  = _disp_data["grip"]
                ts    = _disp_data["ts"]

            if ts != last_ts and ext is not None and wrist is not None:
                last_ts = ts

                ext_r   = _safe_resize_rgb(ext, IMG_EXPECTED_SIZE)
                wrist_r = _safe_resize_rgb(wrist, IMG_EXPECTED_SIZE)
                ext_bgr   = cv2.cvtColor(ext_r, cv2.COLOR_RGB2BGR)
                wrist_bgr = cv2.cvtColor(wrist_r, cv2.COLOR_RGB2BGR)

                pad = np.zeros((ext_bgr.shape[0], 10, 3), dtype=np.uint8)
                mosaic = np.hstack([ext_bgr, pad, wrist_bgr])

                cv2.putText(mosaic, "Exterior", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(mosaic, "Wrist", (ext_bgr.shape[1] + 20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

                _draw_overlay(mosaic, prompt, joint, grip)

                cv2.imshow(WINDOW_NAME, mosaic)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("Display thread: window closed by user.")
                break

        except Exception as e:
            print(f"Display thread error: {e}")
            time.sleep(0.05)

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


# ---------- FastAPI schema ----------
class RequestBody(BaseModel):
    exterior_image_1_left: str   # base64 PNG/JPEG (RGB)
    wrist_image_left: str        # base64 (RGB)
    joint_position: List[float]  # len=7
    gripper_position: List[float] # len=1
    prompt: str

class ResponseBody(BaseModel):
    actions: List[List[float]]   # (T, 8)


# ---------- load policy once ----------
_cfg = _config.get_config("pi0_fast_droid")
_ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
_policy = policy_config.create_trained_policy(_cfg, _ckpt)

app = FastAPI(title="Pi-0 FAST Droid Inference")


def b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


@app.on_event("startup")
def _startup_log():
    print("Server startup complete.")
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    if SHOW_WINDOW:
        th = threading.Thread(target=_display_loop, daemon=True)
        th.start()
        print("Display thread launched.")
    else:
        if not HAS_DISPLAY:
            print("Display disabled: no DISPLAY in environment.")
        else:
            print("Display disabled: SHOW_WINDOW=0.")


@app.on_event("shutdown")
def _shutdown_cleanup():
    global _stop_display
    _stop_display = True
    print("Shutdown: display thread stop requested.")


@app.post("/api/infer", response_model=ResponseBody)
def infer(body: RequestBody):
    t0 = time.time()
    try:
        if len(body.joint_position) != 7:
            raise HTTPException(status_code=400, detail="joint_position must have length 7")
        if len(body.gripper_position) != 1:
            raise HTTPException(status_code=400, detail="gripper_position must have length 1")

        ext_img_pil   = b64_to_pil(body.exterior_image_1_left)
        wrist_img_pil = b64_to_pil(body.wrist_image_left)

        ext_img   = np.array(ext_img_pil)   # RGB
        wrist_img = np.array(wrist_img_pil) # RGB
        joint     = np.array(body.joint_position,   dtype=np.float32)
        grip      = np.array(body.gripper_position, dtype=np.float32)

        print("Request received.")
        print("Prompt:", (body.prompt[:200] + ("..." if len(body.prompt) > 200 else "")))
        print("Exterior image shape:", ext_img.shape, "dtype:", ext_img.dtype)
        print("Wrist image shape:", wrist_img.shape, "dtype:", wrist_img.dtype)
        print("Joint shape:", joint.shape, "values:", np.array2string(joint, precision=4))
        print("Gripper shape:", grip.shape, "values:", np.array2string(grip, precision=4))

        if ext_img.shape[:2] != IMG_EXPECTED_SIZE:
            print(f"Warning: exterior image size is {ext_img.shape[:2]}, expected {IMG_EXPECTED_SIZE}.")
        if wrist_img.shape[:2] != IMG_EXPECTED_SIZE:
            print(f"Warning: wrist image size is {wrist_img.shape[:2]}, expected {IMG_EXPECTED_SIZE}.")

        # ----- save received images -----
        ts_str = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        exterior_path = os.path.join(SAVE_DIR, f"exterior_{ts_str}.png")
        wrist_path    = os.path.join(SAVE_DIR, f"wrist_{ts_str}.png")
        try:
            # ext_img, wrist_img are RGB; convert to BGR for cv2.imwrite or save via PIL
            cv2.imwrite(exterior_path, cv2.cvtColor(ext_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(wrist_path,    cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))
            print("Saved images:", exterior_path, wrist_path)
        except Exception as e:
            print("Failed to save images:", e)

        # ----- update display buffer -----
        if SHOW_WINDOW:
            with _disp_lock:
                _disp_data["ext"]    = ext_img
                _disp_data["wrist"]  = wrist_img
                _disp_data["prompt"] = body.prompt
                _disp_data["joint"]  = joint.copy()
                _disp_data["grip"]   = float(grip[0])
                _disp_data["ts"]     = time.time()
                _disp_ready_event.set()

        example = {
            "observation/exterior_image_1_left":  ext_img,
            "observation/wrist_image_left":       wrist_img,
            "observation/joint_position":         joint,
            "observation/gripper_position":       grip,
            "prompt":                              body.prompt,
        }

        t1 = time.time()
        with torch.inference_mode():
            result = _policy.infer(example)  # expected: dict with "actions"
        t2 = time.time()

        if "actions" not in result:
            raise HTTPException(status_code=500, detail="Policy returned no 'actions' field")

        actions_arr = np.array(result["actions"])
        # shape normalization: (T,8) | (1,T,8) | (1,8)
        if actions_arr.ndim == 3:
            actions_arr = actions_arr[0]
        elif actions_arr.ndim == 2:
            pass
        elif actions_arr.ndim == 1 and actions_arr.size == 8:
            actions_arr = actions_arr.reshape(1, 8)
        else:
            raise HTTPException(status_code=500, detail=f"Unexpected actions shape: {actions_arr.shape}")

        if actions_arr.shape[1] != 8:
            raise HTTPException(status_code=500, detail=f"Actions last dimension must be 8, got {actions_arr.shape}")

        print("Inference done. Latency: preprocess=%.3fs, infer=%.3fs, total=%.3fs"
              % (t1 - t0, t2 - t1, time.time() - t0))
        print("Actions shape:", actions_arr.shape)
        if actions_arr.shape[0] > 0:
            print("First action sample:", np.array2string(actions_arr[0], precision=4))

        actions_list: List[List[float]] = actions_arr.astype(float).tolist()
        return {"actions": actions_list}

    except HTTPException:
        raise
    except Exception as e:
        print("Unhandled error during inference:", repr(e))
        raise HTTPException(status_code=400, detail=str(e))
