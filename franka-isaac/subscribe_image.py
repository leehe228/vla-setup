#!/usr/bin/env python3
import rclpy                                                             
from rclpy.node import Node                                             
from sensor_msgs.msg import Image                                       
from cv_bridge import CvBridge                                          
import cv2                                                              

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()                                         
        # '/rgb' 대신 실제 사용 중인 토픽명을 넣을 수도 있습니다
        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning
        self.get_logger().info('Subscribed to /rgb')                    

    def listener_callback(self, msg: Image):
        # ROS Image -> OpenCV 이미지로 변환  [oai_citation:0‡wiki.ros.org](https://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython?utm_source=chatgpt.com)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        # 윈도우에 표시  [oai_citation:1‡Automatic Addison](https://automaticaddison.com/getting-started-with-opencv-in-ros-2-foxy-fitzroy-python/?utm_source=chatgpt.com)
        cv2.imshow('Camera View', cv_image)
        cv2.waitKey(1)  # 1ms 대기: 키 입력 처리 및 프레임 갱신

def main(args=None):
    rclpy.init(args=args)                                               
    node = ImageSubscriber()                                            
    try:
        rclpy.spin(node)                                                
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 OpenCV 윈도우 정리  [oai_citation:2‡Medium](https://ibrahimmansur4.medium.com/integrating-opencv-with-ros2-a-comprehensive-guide-to-computer-vision-in-robotics-66b97fa2de92?utm_source=chatgpt.com)
        cv2.destroyAllWindows()                                         
        node.destroy_node()                                            
        rclpy.shutdown()                                               

if __name__ == '__main__':
    main()
