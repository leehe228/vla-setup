#!/usr/bin/env python3
"""
Auto-detect and open the first working USB webcam on Linux.
Press 'q' to quit.
"""

import glob                                             # 표준 라이브러리
import cv2                                              # pip install opencv-python
import sys

def find_working_camera(max_index: int = 10):
    """
    /dev/video*를 순회하며 VideoCapture가 성공하는 첫 인덱스를 반환합니다.
    """
    # /dev/video0, /dev/video1, ... 을 glob으로 수집
    devs = sorted(glob.glob('/dev/video*'))
    for dev in devs:
        try:
            idx = int(dev.replace('/dev/video',''))
        except ValueError:
            continue
        cap = cv2.VideoCapture(idx)                     # 인덱스 시도
        if cap.isOpened():
            cap.release()
            print(f"✅  Found working camera: index={idx} ({dev})")
            return idx
    print("❌  No working camera found.")
    return None

def main():
    cam_index = find_working_camera()
    if cam_index is None:
        sys.exit(1)

    # 발견된 카메라 인덱스로 비디오 캡처 시작
    cap = cv2.VideoCapture(cam_index)
    print(f"▶️  Opening camera at index {cam_index}. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Frame grab failed; exiting.")
            break

        cv2.imshow(f"Webcam Index {cam_index}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
