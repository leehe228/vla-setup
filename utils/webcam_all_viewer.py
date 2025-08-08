#!/usr/bin/env python3
"""
Detect and display *all* working USB webcams on Linux.
Press 'q' in *any* window (or Ctrl-C in the terminal) to quit.
"""

import glob
import cv2
import sys

def find_working_cameras() -> list[int]:
    """
    /dev/video* 장치를 모두 확인하여 VideoCapture가 열리는 인덱스 리스트를 반환.
    """
    indices = []
    for dev in sorted(glob.glob("/dev/video*")):          # e.g. /dev/video0 …
        try:
            idx = int(dev.replace("/dev/video", ""))
        except ValueError:
            continue

        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)         # V4L2 백엔드 권장
        if cap.isOpened():
            indices.append(idx)
            cap.release()
    return indices


def main():
    cam_indices = find_working_cameras()
    if not cam_indices:
        print("❌  No working camera found.")
        sys.exit(1)

    print(f"✅  Found cameras: {cam_indices}")

    # 모든 카메라 열기
    caps      = [cv2.VideoCapture(i, cv2.CAP_V4L2) for i in cam_indices]
    win_names = [f"Webcam {i}" for i in cam_indices]
    for name in win_names:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)          # 창 크기 조절 가능

    try:
        while True:
            for cap, name in zip(caps, win_names):
                ret, frame = cap.read()
                if not ret:
                    print(f"⚠️  Frame grab failed on {name}; skipping.")
                    continue
                cv2.imshow(name, frame)

            # 아무 창에서나 'q' 입력 시 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
