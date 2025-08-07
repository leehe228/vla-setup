import base64, io, numpy as np, torch
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openpi.models import model as _model                 # these imports come from openpi
from openpi.policies import droid_policy, policy_config
from openpi.shared import download
from openpi.training import config as _config

# ---------- FastAPI schema ----------
class RequestBody(BaseModel):
    exterior_image_1_left: str   # base64 PNG/JPEG
    wrist_image_left: str        # base64
    joint_position: list[float]  # len=7
    gripper_position: list[float] # len=1
    prompt: str

class ResponseBody(BaseModel):
    actions: list[float]         # len=8

# ---------- load policy once ----------
_cfg = _config.get_config("pi0_fast_droid")
_ckpt = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
_policy = policy_config.create_trained_policy(_cfg, _ckpt)

app = FastAPI(title="Pi-0 FAST Droid Inference")

def b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")

@app.post("/api/infer", response_model=ResponseBody)
def infer(body: RequestBody):
    try:
        ext_img  = np.array(b64_to_pil(body.exterior_image_1_left))
        wrist_img= np.array(b64_to_pil(body.wrist_image_left))
        joint    = np.array(body.joint_position,  dtype=np.float32)
        grip     = np.array(body.gripper_position,dtype=np.float32)

        example = {
            "observation/exterior_image_1_left":  ext_img,
            "observation/wrist_image_left":       wrist_img,
            "observation/joint_position":         joint,
            "observation/gripper_position":       grip,
            "prompt":                             body.prompt,
        }
        result = _policy.infer(example)           # returns dict with "actions"
        actions = result["actions"][0].tolist()   # shape (1,8)→list[8]
        return {"actions": actions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
