import base64
import io
import time
import logging
from typing import List

import numpy as np
import torch
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

# ---------- FastAPI schema ----------
class RequestBody(BaseModel):
    exterior_image_1_left: str   # base64 PNG/JPEG
    wrist_image_left: str        # base64
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


def _np_shape(x) -> str:
    try:
        return str(np.asarray(x).shape)
    except Exception:
        return "unknown"


@app.on_event("startup")
def _startup_log():
    log.info("Server startup complete.")
    log.info("Torch version: %s", torch.__version__)
    log.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info("CUDA device: %s", torch.cuda.get_device_name(0))


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

        ext_img   = np.array(ext_img_pil)   # HxWx3 uint8
        wrist_img = np.array(wrist_img_pil) # HxWx3 uint8
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
        if ext_img.shape[:2] != (244, 244):
            log.warning("Exterior image size is %s, expected (244, 244).", ext_img.shape[:2])
        if wrist_img.shape[:2] != (244, 244):
            log.warning("Wrist image size is %s, expected (244, 244).", wrist_img.shape[:2])

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
            # assume (B, T, D) -> take batch 0
            actions_arr = actions_arr[0]
        elif actions_arr.ndim == 2:
            # assume (T, D) or (1, D)
            pass
        elif actions_arr.ndim == 1 and actions_arr.size == 8:
            actions_arr = actions_arr.reshape(1, 8)
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected actions shape: {actions_arr.shape}"
            )

        # Final checks
        if actions_arr.shape[1] != 8:
            raise HTTPException(
                status_code=500,
                detail=f"Actions last dimension must be 8, got {actions_arr.shape}"
            )

        # ---- response logging ----
        log.info("Inference done. Latency: preprocess=%.3fs, infer=%.3fs, total=%.3fs",
                 (t1 - t0), (t2 - t1), (time.time() - t0))
        log.info("Actions shape: %s", actions_arr.shape)
        if actions_arr.shape[0] > 0:
            log.info("First action sample: %s",
                     np.array2string(actions_arr[0], precision=4))

        actions_list: List[List[float]] = actions_arr.astype(float).tolist()
        return {"actions": actions_list}

    except HTTPException:
        # already a clean HTTP error
        raise
    except Exception as e:
        log.exception("Unhandled error during inference.")
        raise HTTPException(status_code=400, detail=str(e))
