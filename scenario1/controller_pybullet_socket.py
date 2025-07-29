#!/usr/bin/env python3
"""
PyBullet controller (socket protocol with length‑prefix)
"""
import socket, struct, time, math, cv2, numpy as np
import pybullet as p, pybullet_data

HOST, PORT = "127.0.0.1", 5555
HEADER_SZ  = 4
STEP_TIME  = 1/240.0
CYCLES     = 100

def recvall(conn, until_newline=False):
    buf = b''
    while True:
        chunk = conn.recv(32)
        if not chunk:
            return None
        buf += chunk
        if until_newline and b'\n' in buf:
            line, rest = buf.split(b'\n', 1)
            return line.decode().strip(), rest
        return buf

def connect_robot():
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-9.81)
    p.loadURDF("plane.urdf")
    robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
    p.setTimeStep(STEP_TIME); p.setRealTimeSimulation(0)
    return robot

def get_camera():
    v = p.computeViewMatrix([0.5,0,0.5],[0,0,0.3],[0,0,1])
    P = p.computeProjectionMatrixFOV(60,1,0.1,2)
    _,_,rgb,_,_ = p.getCameraImage(128,128,viewMatrix=v,projectionMatrix=P)
    return np.reshape(rgb,(128,128,4)).astype(np.uint8)

def main():
    # socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT)); print("[Controller] Connected")

    robot = connect_robot(); ee = 11

    for i in range(CYCLES):
        # JPEG encode camera
        jpg = cv2.imencode(".jpg", get_camera())[1]
        sock.sendall(struct.pack('>I', len(jpg)) + jpg.tobytes())

        # receive action line (newline‑terminated)
        line = b''
        while b'\n' not in line:
            chunk = sock.recv(32)
            if not chunk:
                raise ConnectionAbortedError("Socket closed")
            line += chunk
        action = line.decode().strip()
        dx,dy,dz,drx,dry,drz,grip = map(int, action.split(','))
        print(f"[Controller] Step {i}: {action}")

        pos,orn = p.getLinkState(robot,ee)[4:6]
        eul     = p.getEulerFromQuaternion(orn)
        tgt_pos = [pos[0]+dx*1e-3, pos[1]+dy*1e-3, pos[2]+dz*1e-3]
        tgt_eul = [eul[0]+math.radians(drx*0.1),
                   eul[1]+math.radians(dry*0.1),
                   eul[2]+math.radians(drz*0.1)]
        tgt_orn = p.getQuaternionFromEuler(tgt_eul)
        joints  = p.calculateInverseKinematics(robot, ee, tgt_pos, tgt_orn)

        for j in range(7):
            p.setJointMotorControl2(robot,j,p.POSITION_CONTROL,joints[j])
        if grip==1:
            p.setJointMotorControl2(robot,9,p.POSITION_CONTROL,0.04)
            p.setJointMotorControl2(robot,10,p.POSITION_CONTROL,0.04)
        elif grip==-1:
            p.setJointMotorControl2(robot,9,p.POSITION_CONTROL,0.0)
            p.setJointMotorControl2(robot,10,p.POSITION_CONTROL,0.0)

        for _ in range(100): p.stepSimulation(); time.sleep(STEP_TIME)

    sock.close(); p.disconnect(); print("[Controller] Done")

if __name__ == "__main__":
    main()
