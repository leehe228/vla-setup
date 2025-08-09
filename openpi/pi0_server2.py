import os
import base64
import io
import time
import logging
import threading
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

# ---------- logging setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("pi0-fast-droid-server")

# ---------- display config ----------
# 창 표시 여부: SHOW_WINDOW=0 이거나 DISPLAY 미설정이면 비활성화
SHOW_WINDOW_ENV = os.environ.get("SHOW_WINDOW", "1") == "1"
HAS_DISPLAY = bool(os.environ.get("DISPLAY"))
SHOW_WINDOW = SHOW_WINDOW_ENV and HAS_DISPLAY
WINDOW_NAME = "Pi0 Request Monitor"

IMG_EXPECTED_SIZE = (244, 244)

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
_disp_thread_started = False
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
    """수신 이미지가 생기면 창 생성→갱신 루프."""
    global _disp_thread_started, _stop_display
    _disp_thread_started = True

    # 첫 프레임이 들어올 때까지 대기 (최대 10초)
    if not _disp_ready_event.wait(timeout=10.0):
        log.warning("Display thread timeout waiting for first frame. Exiting display thread.")
        return

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1000, 520)
    except Exception as e:
        log.warning("OpenCV window cannot be created (headless?). Display disabled. %s", e)
        return

    log.info("Display thread started.")
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

            # 새로운 프레임이 있을 때만 갱신
            if ts != last_ts and ext is not None and wrist is not None:
                last_ts = ts

                # ensure expected size (244x244), then RGB->BGR 변환
                ext_r   = _safe_resize_rgb(ext, IMG_EXPECTED_SIZE)
                wrist_r = _safe_resize_rgb(wrist, IMG_EXPECTED_SIZE)
                ext_bgr   = cv2.cvtColor(ext_r, cv2.COLOR_RGB2BGR)
                wrist_bgr = cv2.cvtColor(wrist_r, cv2.COLOR_RGB2BGR)

                # 좌우 배치
                pad = np.zeros((ext_bgr.shape[0], 10, 3), dtype=np.uint8)
                mosaic = np.hstack([ext_bgr, pad, wrist_bgr])

                # 라벨
                cv2.putText(mosaic, "Exterior", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(mosaic, "Wrist",    (ext_bgr.shape[1] + 10 + 10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

                # 텍스트 오버레이
                _draw_overlay(mosaic, prompt, joint, grip)

                cv2.imshow(WINDOW_NAME, mosaic)

            # 키 입력 폴링
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                log.info("Display window closed by user.")
                break

        except Exception as e:
            log.warning("Display loop error: %s", e)
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
    # return full sequence: shape (T, 8) -> list of list
    actions: List[List[float]]

# ---------- load policy once ----------
_cfg = _config.get_config("pi0_fast_droid")
_ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
_policy = policy_config.create_trained_policy(_cfg, _ckpt)

app = FastAPI(title="Pi-0 FAST Droid Inference")


def b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


@app.on_event("startup")
def _startup_log():
    log.info("Server startup complete.")
    log.info("Torch version: %s", torch.__version__)
    log.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info("CUDA device: %s", torch.cuda.get_device_name(0))

    if SHOW_WINDOW:
        th = threading.Thread(target=_display_loop, daemon=True)
        th.start()
    else:
        if not HAS_DISPLAY:
            log.info("Display disabled: no DISPLAY in environment.")
        else:
            log.info("Display disabled: SHOW_WINDOW=0.")


@app.on_event("shutdown")
def _shutdown_cleanup():
    global _stop_display
    _stop_display = True


@app.post("/api/infer", response_model=ResponseBody)
def infer(body: RequestBody):
    t0 = time.time()
    try:
        # ---- basic validation ----
        if len(body.joint_position) != 7:
            raise HTTPException(status_code=400, detail="joint_position must have length 7")
        if len(body.gripper_position) != 1:
            raise HTTPException(status_code=400, detail="gripper_position must have length 1")

        ext_img_pil   = b64_to_pil(body.exterior_image_1_left)
        wrist_img_pil = b64_to_pil(body.wrist_image_left)

        ext_img   = np.array(ext_img_pil)   # HxWx3 uint8 (RGB)
        wrist_img = np.array(wrist_img_pil) # HxWx3 uint8 (RGB)
        joint     = np.array(body.joint_position,   dtype=np.float32)
        grip      = np.array(body.gripper_position, dtype=np.float32)

        # ---- request logging ----
        log.info("Request received.")
        log.info("Prompt: %s", (body.prompt[:200] + ("..." if len(body.prompt) > 200 else "")))
        log.info("Exterior image shape: %s, dtype: %s", ext_img.shape, ext_img.dtype)
        log.info("Wrist image shape: %s, dtype: %s", wrist_img.shape, wrist_img.dtype)
        log.info("Joint shape: %s, values: %s", joint.shape, np.array2string(joint, precision=4))
        log.info("Gripper shape: %s, values: %s", grip.shape, np.array2string(grip, precision=4))

        # Optional sanity warning for expected size
        if ext_img.shape[:2] != IMG_EXPECTED_SIZE:
            log.warning("Exterior image size is %s, expected %s.", ext_img.shape[:2], IMG_EXPECTED_SIZE)
        if wrist_img.shape[:2] != IMG_EXPECTED_SIZE:
            log.warning("Wrist image size is %s, expected %s.", wrist_img.shape[:2], IMG_EXPECTED_SIZE)

        # ---- update display buffer (첫 프레임 도달 시 event set) ----
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

        # ---- inference ----
        t1 = time.time()
        with torch.inference_mode():
            result = _policy.infer(example)  # expected to contain "actions"
        t2 = time.time()

        if "actions" not in result:
            raise HTTPException(status_code=500, detail="Policy returned no 'actions' field")

        actions_arr = np.array(result["actions"])

        # Robust shape handling:
        # - (T, 8)
        # - (1, T, 8)
        # - (1, 8)  -> promote to (1, 8)
        if actions_arr.ndim == 3:
            actions_arr = actions_arr[0]
        elif actions_arr.ndim == 2:
            pass
        elif actions_arr.ndim == 1 and actions_arr.size == 8:
            actions_arr = actions_arr.reshape(1, 8)
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected actions shape: {actions_arr.shape}"
            )

        if actions_arr.shape[1] != 8:
            raise HTTPException(
                status_code=500,
                detail=f"Actions last dimension must be 8, got {actions_arr.shape}"
            )

        log.info(
            "Inference done. Latency: preprocess=%.3fs, infer=%.3fs, total=%.3fs",
            (t1 - t0), (t2 - t1), (time.time() - t0)
        )
        log.info("Actions shape: %s", actions_arr.shape)
        if actions_arr.shape[0] > 0:
            log.info("First action sample: %s", np.array2string(actions_arr[0], precision=4))

        actions_list: List[List[float]] = actions_arr.astype(float).tolist()
        return {"actions": actions_list}

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Unhandled error during inference.")
        raise HTTPException(status_code=400, detail=str(e))
