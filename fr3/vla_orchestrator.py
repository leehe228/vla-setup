#!/usr/bin/env python3
import os, sys, time, json, base64, io, threading, logging
from typing import Optional, List, Tuple

import cv2
import numpy as np
import requests
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from rcl_interfaces.srv import SetParameters, GetParameters
from rcl_interfaces.msg import Parameter, ParameterValue

from controller_manager_msgs.srv import (
    LoadController, UnloadController, ConfigureController, SwitchController, ListControllers
)

from franka_msgs.action import Move as GripperMove
from franka_msgs.action import Homing as GripperHoming
# 필요시 Grasp도 사용 가능:
# from franka_msgs.action import Grasp as GripperGrasp

# =======================
# 하이퍼파라미터 & 상수
# =======================
SERVER_URL         = "http://ip:port/api/infer"
NAMESPACE          = "fr3"

CAM_FRONT_INDEX    = 0
CAM_WRIST_INDEX    = 10
IMG_SIZE           = 244

CONTROLLER_NAME    = "move_to_goal"
CM_PREFIX          = f"/{NAMESPACE}/controller_manager"
CTRL_NODE_PREFIX   = f"/{NAMESPACE}/{CONTROLLER_NAME}"

# k개 액션만 실행
K_ACTIONS          = 10
# 그리퍼 이동 속도(기본값). 프랑카 그리퍼 Move.goal = {width[m], speed[m/s]}
GRIPPER_SPEED      = float(os.environ.get("GRIPPER_SPEED", "0.10"))
# move_to_goal speed_scale 기본값(0.05~1.0 권장)
ARM_SPEED_SCALE    = float(os.environ.get("ARM_SPEED_SCALE", "0.20"))

ACTION_SCALE   = 0.5
PER_STEP_CLAMP = 0.10
GRIPPER_STEP   = 0.005

LOG_DIR            = os.environ.get("LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "vla_orchestrator.log"), mode="a", encoding="utf-8")
    ]
)
log = logging.getLogger("vla_orchestrator")


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class FrankaOrchestrator(Node):
    """하나의 노드에서
       - 조인트/그리퍼 상태 구독
       - move_to_goal 파라미터 갱신
       - 컨트롤러 로드/설정/스위치
       - 그리퍼 액션 호출
       을 모두 처리
    """
    def __init__(self):
        super().__init__("vla_orchestrator_node")

        # --- 상태 구독 ---
        self.arm_joint_names: List[str] = [f"{NAMESPACE}_joint{i}" for i in range(1, 8)]
        self._arm_js = None        # np.ndarray, shape (7,)
        self._grip_js = None       # np.ndarray, raw finger joint positions (len 1 or 2)
        self._lock = threading.Lock()

        self.create_subscription(JointState, f"/{NAMESPACE}/joint_states", self._on_arm_js, 10)
        self.create_subscription(JointState, f"/{NAMESPACE}/franka_gripper/joint_states", self._on_grip_js, 10)

        # --- 파라미터 서비스 클라이언트 (컨트롤러 노드) ---
        self._set_params_cli = self.create_client(SetParameters, f"{CTRL_NODE_PREFIX}/set_parameters")
        self._get_params_cli = self.create_client(GetParameters, f"{CTRL_NODE_PREFIX}/get_parameters")

        # --- controller_manager 서비스 클라이언트 ---
        self._load_cli      = self.create_client(LoadController,        f"{CM_PREFIX}/load_controller")
        self._unload_cli    = self.create_client(UnloadController,      f"{CM_PREFIX}/unload_controller")
        self._configure_cli = self.create_client(ConfigureController,   f"{CM_PREFIX}/configure_controller")
        self._switch_cli    = self.create_client(SwitchController,      f"{CM_PREFIX}/switch_controller")
        self._list_cli      = self.create_client(ListControllers,       f"{CM_PREFIX}/list_controllers")

        # --- Gripper 액션 클라이언트 ---
        self._grip_move_ac  = ActionClient(self, GripperMove,  f"/{NAMESPACE}/franka_gripper/move")
        self._grip_home_ac  = ActionClient(self, GripperHoming, f"/{NAMESPACE}/franka_gripper/homing")
        # 필요시 Grasp:
        # self._grip_grasp_ac = ActionClient(self, GripperGrasp, f"/{NAMESPACE}/franka_gripper/grasp")

    # ---------------------------
    # 상태 콜백 / 읽기 메서드
    # ---------------------------
    def _on_arm_js(self, msg: JointState):
        # 이름 정렬하여 7개만 뽑음
        idx = []
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        ok = True
        for n in self.arm_joint_names:
            if n not in name_to_idx:
                ok = False
                break
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

    def get_latest_state(self, wait_sec: float = 2.0) -> Tuple[np.ndarray, float]:
        """팔 조인트(7,), 그리퍼 폭(width[m])을 반환.
        그리퍼 width는 finger joint 포지션에서 유도:
        - 보통 두 손가락 조인트는 동일하며 합이 width(혹은 각 0~0.04, width는 2*pos)로 사용됨.
        - 안전하게: len==2면 sum, len==1이면 2*pos 로 환산.
        """
        t0 = time.time()
        arm, grip = None, None
        while time.time() - t0 < wait_sec:
            with self._lock:
                arm = None if self._arm_js is None else self._arm_js.copy()
                gj  = None if self._grip_js is None else self._grip_js.copy()
            if arm is not None and gj is not None:
                # width 추정
                if len(gj) >= 2:
                    width = float(np.clip(gj[0] + gj[1], 0.0, 0.08))
                else:
                    width = float(np.clip(2.0 * gj[0], 0.0, 0.08))
                return arm, width
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError("Timed out waiting for joint/gripper states.")

    # ---------------------------
    # 컨트롤러 관리
    # ---------------------------
    def wait_service(self, cli, name, timeout=5.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"Service not available: {name}")

    def load_controller(self, name: str):
        self.wait_service(self._load_cli, "load_controller")
        req = LoadController.Request()
        req.name = name
        fut = self._load_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        resp = fut.result()
        if not (resp and resp.ok):
            raise RuntimeError(f"load_controller({name}) failed")
        log.info(f"Loaded controller: {name}")

    def configure_controller(self, name: str):
        self.wait_service(self._configure_cli, "configure_controller")
        req = ConfigureController.Request()
        req.name = name
        fut = self._configure_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        resp = fut.result()
        if not (resp and resp.ok):
            raise RuntimeError(f"configure_controller({name}) failed")
        log.info(f"Configured controller: {name}")

    def switch_controller(self, start: List[str], stop: List[str], strict=True, timeout_s: float = 5.0):
        self.wait_service(self._switch_cli, "switch_controller")
        req = SwitchController.Request()
        req.activate_controllers   = start
        req.deactivate_controllers = stop
        req.strictness = SwitchController.Request.STRICT if strict else SwitchController.Request.BEST_EFFORT
        req.timeout = rclpy.duration.Duration(seconds=timeout_s).to_msg()
        fut = self._switch_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        resp = fut.result()
        if not (resp and resp.ok):
            raise RuntimeError(f"switch_controller(start={start}, stop={stop}) failed")
        if start:
            log.info(f"Activated: {start}")
        if stop:
            log.info(f"Deactivated: {stop}")

    # ---------------------------
    # move_to_goal 파라미터 I/O
    # ---------------------------
    def set_move_to_goal(self, q_goal: List[float], speed_scale: float):
        self.wait_service(self._set_params_cli, "set_parameters")
        params = [
            Parameter(name="q_goal_",     value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE_ARRAY, double_array_value=list(q_goal))),
            Parameter(name="speed_scale", value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,       double_value=float(speed_scale))),
            Parameter(name="process_finished", value=ParameterValue(type=ParameterType.PARAMETER_BOOL,    bool_value=False)),
        ]
        req = SetParameters.Request(parameters=params)
        fut = self._set_params_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        resp = fut.result()
        if not resp or not all(r.successful for r in resp.results):
            raise RuntimeError("set_parameters on move_to_goal failed")
        log.info(f"Set move_to_goal: speed_scale={speed_scale:.3f}, q_goal={np.array(q_goal)}")

    def wait_process_finished(self, timeout_s: float = 10.0) -> bool:
        self.wait_service(self._get_params_cli, "get_parameters")
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            req = GetParameters.Request(names=["process_finished"])
            fut = self._get_params_cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut)
            resp = fut.result()
            if resp and resp.values and resp.values[0].type == ParameterType.PARAMETER_BOOL:
                if resp.values[0].bool_value:
                    return True
            time.sleep(0.05)
        return False

    # ---------------------------
    # 그리퍼 제어 (Move/Homing)
    # ---------------------------
    def wait_action_server(self, ac: ActionClient, name: str, timeout=5.0):
        if not ac.wait_for_server(timeout_sec=timeout):
            raise RuntimeError(f"Gripper action server not available: {name}")

    def gripper_move(self, width_m: float, speed_mps: float = GRIPPER_SPEED) -> bool:
        self.wait_action_server(self._grip_move_ac, "move")
        goal = GripperMove.Goal()
        goal.width = float(clamp(width_m, 0.0, 0.08))        # Panda gripper typical range ~0.0–0.08 m (각 액션 필드 정의는 franka_msgs 참조)
        goal.speed = float(clamp(speed_mps, 0.01, 0.2))
        fut = self._grip_move_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        handle = fut.result()
        if not handle:
            log.error("Failed to send gripper Move goal")
            return False
        res_fut = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        result = res_fut.result()
        ok = bool(result and result.result and result.result.success)
        log.info(f"Gripper Move to width={goal.width:.3f} m @ {goal.speed:.2f} m/s → success={ok}")
        return ok

    def gripper_homing(self) -> bool:
        self.wait_action_server(self._grip_home_ac, "homing")
        goal = GripperHoming.Goal()
        fut = self._grip_home_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        handle = fut.result()
        if not handle:
            log.error("Failed to send gripper Homing goal")
            return False
        res_fut = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        result = res_fut.result()
        ok = bool(result and result.result and result.result.success)
        log.info(f"Gripper Homing → success={ok}")
        return ok


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {index}")
    return cap


def grab_244(cap: cv2.VideoCapture) -> np.ndarray:
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to grab frame")
    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized[:, :, ::-1]  # BGR→RGB


def post_to_server(ext_img: np.ndarray, wrist_img: np.ndarray,
                   joints: np.ndarray, grip_width: float,
                   prompt: str) -> np.ndarray:
    ext_b64   = pil_to_b64(Image.fromarray(ext_img))
    wrist_b64 = pil_to_b64(Image.fromarray(wrist_img))
    payload = {
        "exterior_image_1_left": ext_b64,
        "wrist_image_left":      wrist_b64,
        "joint_position":        [float(x) for x in joints.tolist()],
        "gripper_position":      [float(grip_width)],
        "prompt":                prompt,
    }
    t0 = time.time()
    resp = requests.post(SERVER_URL, json=payload, timeout=30)
    resp.raise_for_status()
    out = resp.json()
    actions = np.array(out["actions"], dtype=np.float32)  # shape (10, 8)
    log.info(f"VLA response in {time.time()-t0:.3f}s: actions.shape={actions.shape}")
    # 로깅 저장
    npy_path = os.path.join(LOG_DIR, f"actions_{int(time.time())}.npy")
    np.save(npy_path, actions)
    log.info(f"Saved actions to {npy_path}")
    return actions


def ensure_controller_ready(node: FrankaOrchestrator):
    """처음 한 번만: move_to_goal 로드→구성(inactive)까지 보장"""
    # 이미 로드돼있을 수 있으므로 load 실패는 예외로, configure까지 수행
    try:
        node.load_controller(CONTROLLER_NAME)
    except Exception as e:
        log.warning(f"load_controller skipped or failed (maybe already loaded): {e}")

    try:
        node.configure_controller(CONTROLLER_NAME)
    except Exception as e:
        log.warning(f"configure_controller skipped or failed (maybe already configured): {e}")


def main():
    rclpy.init()
    node = FrankaOrchestrator()

    # 카메라 오픈
    cap_front = open_camera(CAM_FRONT_INDEX)
    cap_wrist = open_camera(CAM_WRIST_INDEX)
    log.info(f"Opened cameras: front={CAM_FRONT_INDEX}, wrist={CAM_WRIST_INDEX}")

    # 컨트롤러 준비
    ensure_controller_ready(node)

    # 필요시 한 번 homing
    try:
        node.gripper_homing()
    except Exception as e:
        log.warning(f"Gripper homing skipped: {e}")

    # 프롬프트 최초 입력
    base_prompt = input("초기 프롬프트를 입력하세요 (예: '노란 박스를 공 옆으로 옮겨'): ").strip()
    if not base_prompt:
        base_prompt = "Move the yellow box next to the ball."

    step = 0
    try:
        while True:
            step += 1
            log.info(f"===== ITERATION {step} =====")

            # 1) 상태 & 이미지 수집
            arm_q, grip_w = node.get_latest_state(wait_sec=3.0)
            ext_img  = grab_244(cap_front)
            wrist_img= grab_244(cap_wrist)

            # 2) 프롬프트 입력(빈 입력이면 유지)
            new_prompt = input("[프롬프트] (엔터=유지): ").strip()
            if new_prompt:
                base_prompt = new_prompt

            # 3) 서버로 전송 → 액션 토큰 수신
            actions = post_to_server(ext_img, wrist_img, arm_q, grip_w, base_prompt)
            if actions.shape != (10, 8):
                log.error(f"Unexpected action shape: {actions.shape}")
                continue
            
            q_curr = arm_q.copy()
            grip_curr = float(grip_w)

            # 4) k개만 실행(각 단계마다 사용자 확인)
            # for i in range(min(K_ACTIONS, actions.shape[0])):
            #     act = actions[i]
            #     q_goal = act[:7].astype(float).tolist()
            #     grip_t = float(act[7])

            #     log.info(f"[ACTION {i+1}/{K_ACTIONS}] q_goal={np.array(q_goal)}  grip_target={grip_t:.4f} m")

            #     _ = input("이 액션을 실행할까요? (엔터=실행 / ctrl+c=중단): ")

            #     # 4-1) 그리퍼 먼저
            #     try:
            #         node.gripper_move(width_m=clamp(grip_t, 0.0, 0.08), speed_mps=GRIPPER_SPEED)
            #     except Exception as e:
            #         log.error(f"Gripper move error: {e}")

            #     # 4-2) 팔 이동: inactive → set params → active
            #     try:
            #         # 비활성화
            #         node.switch_controller(start=[], stop=[CONTROLLER_NAME], strict=True, timeout_s=5.0)
            #     except Exception as e:
            #         log.warning(f"Deactivate skipped/failed: {e}")

            #     # 파라미터 적용
            #     try:
            #         node.set_move_to_goal(q_goal=q_goal, speed_scale=ARM_SPEED_SCALE)
            #     except Exception as e:
            #         log.error(f"set_move_to_goal failed: {e}")
            #         continue

            #     # 활성화(이때 move_to_goal이 on_activate에서 최신 파라미터로 궤적 생성)
            #     try:
            #         node.switch_controller(start=[CONTROLLER_NAME], stop=[], strict=True, timeout_s=5.0)
            #     except Exception as e:
            #         log.error(f"Activate failed: {e}")
            #         continue

            #     # 완료 대기 (컨트롤러가 process_finished=true로 세팅)
            #     finished = node.wait_process_finished(timeout_s=12.0)
            #     log.info(f"Arm motion finished={finished}")
            
            for i in range(min(K_ACTIONS, actions.shape[0])):
                act = actions[i].astype(float)
                dq_raw   = ACTION_SCALE * act[:7]              # (7,) Δrad
                dq       = np.clip(dq_raw, -PER_STEP_CLAMP, +PER_STEP_CLAMP)
                d_grip   = float(act[7]) * GRIPPER_STEP        # Δm

                q_goal   = (q_curr + dq).tolist()
                grip_goal= clamp(grip_curr + d_grip, 0.0, 0.08)

                log.info(
                    "[ACTION %d/%d]\n"
                    "  q_curr= %s\n"
                    "  Δq    = %s (clamped by %.3f)\n"
                    "  q_goal= %s\n"
                    "  grip_curr= %.4f m, Δgrip= %.4f m -> grip_goal= %.4f m",
                    i+1, K_ACTIONS,
                    np.array2string(q_curr, precision=4),
                    np.array2string(dq,    precision=4), PER_STEP_CLAMP,
                    np.array2string(np.array(q_goal), precision=4),
                    grip_curr, d_grip, grip_goal
                )

                _ = input("이 액션을 실행할까요? (엔터=실행 / ctrl+c=중단): ")

                # 1) 그리퍼 먼저
                try:
                    node.gripper_move(width_m=grip_goal, speed_mps=GRIPPER_SPEED)
                except Exception as e:
                    log.error(f"Gripper move error: {e}")

                # 2) 팔 이동: inactive → set params → active
                try:
                    node.switch_controller(start=[], stop=[CONTROLLER_NAME], strict=True, timeout_s=5.0)
                except Exception as e:
                    log.warning(f"Deactivate skipped/failed: {e}")

                try:
                    node.set_move_to_goal(q_goal=q_goal, speed_scale=ARM_SPEED_SCALE)
                except Exception as e:
                    log.error(f"set_move_to_goal failed: {e}")
                    continue

                try:
                    node.switch_controller(start=[CONTROLLER_NAME], stop=[], strict=True, timeout_s=5.0)
                except Exception as e:
                    log.error(f"Activate failed: {e}")
                    continue

                finished = node.wait_process_finished(timeout_s=12.0)
                log.info(f"Arm motion finished={finished}")

                # 3) 실행 후 실제 상태를 다시 읽어 다음 Δ 적용 기준 업데이트
                try:
                    q_curr, grip_curr = node.get_latest_state(wait_sec=2.0)
                except Exception as e:
                    log.warning(f"Failed to refresh state after action: {e}")
                    # 실패해도 로컬 추정치로 계속 진행
                    q_curr   = np.array(q_goal, dtype=float)
                    grip_curr= grip_goal

            # 5) 다음 루프(새 상태/이미지로 다시 추론)

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        cap_front.release()
        cap_wrist.release()
        rclpy.shutdown()


# rcl_interfaces/ParameterType enum alias (간단히 쓸 수 있게)
class ParameterType:
    PARAMETER_NOT_SET   = 0
    PARAMETER_BOOL      = 1
    PARAMETER_INTEGER   = 2
    PARAMETER_DOUBLE    = 3
    PARAMETER_STRING    = 4
    PARAMETER_BYTE_ARRAY= 5
    PARAMETER_BOOL_ARRAY= 6
    PARAMETER_INTEGER_ARRAY=7
    PARAMETER_DOUBLE_ARRAY = 8
    PARAMETER_STRING_ARRAY = 9

if __name__ == "__main__":
    main()
