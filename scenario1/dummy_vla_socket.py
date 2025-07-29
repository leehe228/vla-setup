#!/usr/bin/env python3
"""
Dummy VLA (socket, length‑prefix protocol)
"""
import socket, struct, time, random
import cv2  # Import OpenCV
import numpy as np  # Import NumPy for array handling

HOST, PORT = "127.0.0.1", 5555
HEADER_SZ  = 4                               # uint32 image length
TIME_SLEEP = 0.01

def recvall(conn, n):
    data = b''
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data

def random_action_line():
    vals = [random.randint(-255, 255) for _ in range(6)]
    vals.append(random.choice([-1, 0, 1]))
    return (','.join(map(str, vals)) + '\n').encode()

def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT)); srv.listen(1)
    print(f"[VLA‑Dummy] Listening on {HOST}:{PORT}")
    conn, addr = srv.accept()
    print(f"[VLA‑Dummy] Client {addr} connected")

    try:
        while True:
            hdr = recvall(conn, HEADER_SZ)
            if not hdr:
                break
            img_len = struct.unpack('>I', hdr)[0]
            img_data = recvall(conn, img_len)  # Receive image bytes
            if img_data:
                # Decode the image data to a NumPy array
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                # Decode the image using OpenCV
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    # Display the image in a window
                    cv2.imshow('Received Image', img)
                    # Wait for 1ms and check if 'q' is pressed to exit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            time.sleep(TIME_SLEEP)
            conn.sendall(random_action_line())
    except (KeyboardInterrupt, ConnectionResetError):
        pass
    finally:
        conn.close(); srv.close()
        cv2.destroyAllWindows()  # Close OpenCV windows
        print("[VLA‑Dummy] Shutdown")

if __name__ == "__main__":
    main()
