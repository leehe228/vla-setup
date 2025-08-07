#!/usr/bin/env python3
"""
Read frames from the first USB webcam (index-0) and show them in a window.
Press “q” to quit.  Works on Linux, Windows, macOS if OpenCV is installed.
"""

import cv2                     # pip install opencv-python
import time

def main(cam_index: int = 0):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)   # on Linux just omit CAP_DSHOW

    if not cap.isOpened():
        raise RuntimeError(f"Camera index {cam_index} could not be opened")

    print("Camera opened.  Press  q  in the window to quit.")
    fps_counter, t0 = 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed; retrying …")
            continue

        # optional: resize or process frame here
        cv2.imshow("USB Webcam (press q to quit)", frame)
        fps_counter += 1

        # exit on key-press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # simple FPS display in console
        if fps_counter >= 60:
            t1 = time.time()
            print(f"FPS ≈ {fps_counter/(t1-t0):.1f}")
            fps_counter, t0 = 0, t1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
