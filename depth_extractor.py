#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class DepthPixelExtractor:
    def __init__(self):
        self.bridge = CvBridge()
        
        # Subscribe to the depth image topic
        # The topic name may vary depending on your RealSense configuration
        self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)
        
        # Pixel coordinates to extract (modify these as needed)
        self.pixel_x = 260  # Example: center pixel for 640x480 image
        self.pixel_y = 435
        
        rospy.loginfo("Depth pixel extractor initialized")

    def depth_callback(self, depth_msg):
        try:
            # Convert ROS Image message to OpenCV image
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            
            # Extract depth value at the specified pixel coordinates
            depth_value = depth_image[self.pixel_y, self.pixel_x]
            
            # Convert depth value to meters (depends on your camera and depth encoding)
            # For 16-bit depth images, typical conversion is:
            depth_meters = depth_value * 0.001  # Convert from mm to meters
            
            rospy.loginfo(f"Depth at pixel ({self.pixel_x}, {self.pixel_y}): {depth_value} units, {depth_meters:.3f} meters")
            
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")

if __name__ == '__main__':
    rospy.init_node('depth_pixel_extractor')
    
    extractor = DepthPixelExtractor()
    
    rospy.spin()