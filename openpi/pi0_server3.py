import os
import base64
import io
import time
import threading
import builtins
from collections import deque
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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

# ---------- dashboard buffers ----------
DASHBOARD_HISTORY = int(os.environ.get("DASHBOARD_HISTORY", "50"))     # keep last N requests
DASHBOARD_LOG_LINES = int(os.environ.get("DASHBOARD_LOG_LINES", "300"))# keep last N print lines
RECENTS = deque(maxlen=DASHBOARD_HISTORY)  # list of dicts
LOG_LINES = deque(maxlen=DASHBOARD_LOG_LINES)

# Monkey-patch print: 콘솔 + 메모리 버퍼 동시 기록
_real_print = print
def _print_and_buffer(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    stamped = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    _real_print(stamped, **kwargs)
    try:
        LOG_LINES.append(stamped)
    except Exception:
        pass
builtins.print = _print_and_buffer

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

# 정적 파일(이미지) 서빙
app.mount("/images", StaticFiles(directory=SAVE_DIR), name="images")


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


@app.get("/", response_class=HTMLResponse)
def root():
    # 간단 리다이렉트
    return HTMLResponse('<meta http-equiv="refresh" content="0; url=/dashboard">')


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    # 아주 단순한 폴링 기반 대시보드
    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Pi0 Dashboard</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 0; background:#111; color:#eee; }}
 header {{ padding: 12px 16px; font-weight: 600; background:#222; position:sticky; top:0; }}
 .wrap {{ display:flex; gap:12px; padding: 12px; }}
 .panel {{ background:#1b1b1b; border:1px solid #333; border-radius:10px; padding:12px; flex:1; }}
 img {{ max-width:100%; height:auto; border-radius:8px; border:1px solid #333; }}
 pre {{ white-space:pre-wrap; word-break:break-word; background:#141414; padding:8px; border-radius:8px; border:1px solid #333; max-height:50vh; overflow:auto; }}
 .row {{ display:flex; gap:12px; }}
 .item {{ flex:1; }}
 .meta {{ font-size:13px; color:#bbb; }}
 code {{ color:#9fe3ff; }}
</style>
</head>
<body>
<header>Pi-0 FAST Droid — Live Dashboard</header>
<div class="wrap">
  <div class="panel" style="flex:1.6">
    <h3>Latest Request</h3>
    <div id="meta" class="meta">Waiting for data...</div>
    <div class="row">
      <div class="item">
        <div>Exterior</div>
        <img id="ext" src="" alt="exterior">
      </div>
      <div class="item">
        <div>Wrist</div>
        <img id="wrist" src="" alt="wrist">
      </div>
    </div>
    <h4>First action (if any)</h4>
    <pre id="first_action"></pre>
  </div>
  <div class="panel" style="flex:1">
    <h3>Recent Logs</h3>
    <pre id="logs"></pre>
  </div>
</div>
<script>
async function refresh() {{
  try {{
    const r = await fetch('/api/recent');
    const data = await r.json();

    // logs
    const logs = (data.logs || []).join('\\n');
    document.getElementById('logs').textContent = logs;

    // latest
    const latest = (data.recent && data.recent[0]) || null;
    if (latest) {{
      const meta = `
Time: ${'{'}latest.ts{'}'}\\n
Prompt: ${'{'}latest.prompt || ''{'}'}\\n
Joints: ${'{'}(latest.joint||[]).map(v=>v.toFixed(4)).join(', '){'}'}\\n
Grip: ${'{'}latest.grip || 0{'}'} m\\n
Sequence length: ${'{'}latest.T || 0{'}'}\\n
`;
      document.getElementById('meta').textContent = meta;

      // bust cache with ts
      const ts = Date.now();
      document.getElementById('ext').src = latest.exterior_url + '?t=' + ts;
      document.getElementById('wrist').src = latest.wrist_url + '?t=' + ts;

      const fa = latest.first_action ? JSON.stringify(latest.first_action, null, 2) : '(none)';
      document.getElementById('first_action').textContent = fa;
    }}
  }} catch (e) {{
    console.error(e);
  }} finally {{
    setTimeout(refresh, 1000);
  }}
}}
refresh();
</script>
</body>
</html>
"""
    return HTMLResponse(html)


@app.get("/api/recent", response_class=JSONResponse)
def api_recent():
    # 최근 요청 메타 + 로그 tail 반환
    return JSONResponse({
        "recent": list(RECENTS)[::-1],  # newest first
        "logs": list(LOG_LINES)[-DASHBOARD_LOG_LINES:]
    })


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
        exterior_fname = f"exterior_{ts_str}.png"
        wrist_fname    = f"wrist_{ts_str}.png"
        exterior_path = os.path.join(SAVE_DIR, exterior_fname)
        wrist_path    = os.path.join(SAVE_DIR, wrist_fname)
        try:
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

        # ----- update dashboard RECENTS -----
        try:
            first_action = actions_arr[0].astype(float).tolist() if actions_arr.shape[0] > 0 else None
            RECENTS.append({
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prompt": body.prompt,
                "joint": joint.astype(float).tolist(),
                "grip": float(grip[0]),
                "exterior_url": f"/images/{exterior_fname}",
                "wrist_url":    f"/images/{wrist_fname}",
                "first_action": first_action,
                "T": int(actions_arr.shape[0]),
            })
        except Exception as e:
            print("Failed to update RECENTS:", e)

        actions_list: List[List[float]] = actions_arr.astype(float).tolist()
        return {"actions": actions_list}

    except HTTPException:
        raise
    except Exception as e:
        print("Unhandled error during inference:", repr(e))
        raise HTTPException(status_code=400, detail=str(e))
