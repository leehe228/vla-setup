#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from datetime import datetime

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.listener_callback,
            10
        )
        self.get_logger().info('Subscribed to /rgb')

    def listener_callback(self, msg: Image):
        # ROS Image -> OpenCV 이미지로 변환
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        # 1) 리사이즈 (320x210)
        resized = cv2.resize(cv_image, (320, 210))

        # 2) 타임스탬프 생성 및 우측 상단 위치 계산
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 글자 크기, 두께 설정
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        # 텍스트 크기 측정
        (w, h), _ = cv2.getTextSize(ts, font, font_scale, thickness)
        # 우측 상단 여백 5px
        x = resized.shape[1] - w - 5
        y = h + 5

        # 3) 텍스트 오버레이 (흰색 글자, 검은색 테두리)
        cv2.putText(resized, ts, (x, y), font, font_scale, (0, 0, 0), thickness+1, cv2.LINE_AA)
        cv2.putText(resized, ts, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # 4) 윈도우에 표시
        cv2.imshow('Camera View', resized)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
