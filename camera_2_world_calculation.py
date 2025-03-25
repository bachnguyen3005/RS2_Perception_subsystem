import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def pixel_to_robot_base(pixel_x, pixel_y, depth, camera_intrinsics, camera_extrinsics):
    """
    Transform pixel coordinates to robot base frame coordinates.
    
    Parameters:
    - pixel_x, pixel_y: Pixel coordinates in the image
    - depth: Depth value in meters at the pixel (from depth camera)
    - camera_intrinsics: Dictionary with camera intrinsic parameters
    - camera_extrinsics: Dictionary with camera extrinsic parameters
    
    Returns:
    - (x, y, z): Coordinates in the robot base frame (in meters)
    """
    # Extract intrinsic parameters
    fx = camera_intrinsics['K'][0]
    fy = camera_intrinsics['K'][4]
    cx = camera_intrinsics['K'][2]
    cy = camera_intrinsics['K'][5]
    
    # Extract extrinsic parameters
    translation = np.array([
        camera_extrinsics['translation']['x'],
        camera_extrinsics['translation']['y'],
        camera_extrinsics['translation']['z']
    ])
    
    quaternion = np.array([
        camera_extrinsics['rotation']['x'],
        camera_extrinsics['rotation']['y'],
        camera_extrinsics['rotation']['z'],
        camera_extrinsics['rotation']['w']
    ])
    
    # 1. Convert pixel coordinates to normalized camera coordinates
    x_normalized = (pixel_x - cx) / fx
    y_normalized = (pixel_y - cy) / fy
    
    # 2. Calculate 3D point in camera frame
    point_camera = np.array([
        x_normalized * depth,
        y_normalized * depth,
        depth
    ])
    
    # 3. Convert quaternion to rotation matrix
    # Note: scipy expects quaternion as [w, x, y, z], but our quaternion is [x, y, z, w]
    rot = R.from_quat([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    rotation_matrix = rot.as_matrix()
    
    # 4. Transform point from camera frame to robot base frame
    point_robot = rotation_matrix @ point_camera + translation
    
    
    point_robot[0] = (point_robot [0])
    point_robot[1] = (point_robot [1]+1.05)
    return point_robot

camera_intrinsics = {
    'height': 480,
    'width': 640,
    'distortion_model': 'plumb_bob',
    'D': [0.0, 0.0, 0.0, 0.0, 0.0],
    'K': [606.1439208984375, 0.0, 319.3987731933594, 
          0.0, 604.884033203125, 254.05661010742188, 
          0.0, 0.0, 1.0],
    'R': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    'P': [606.1439208984375, 0.0, 319.3987731933594, 0.0,
          0.0, 604.884033203125, 254.05661010742188, 0.0,
          0.0, 0.0, 1.0, 0.0]
}

camera_extrinsics = {
    'translation': {
        'x': -0.016594289106093278,
        'y': -0.8427389324733021,
        'z': 0.5208564791593558
    },
    'rotation': {
        'x': -0.18820164875470868,
        'y': 0.1854343230394968,
        'z': 0.6803675971177089,
        'w': 0.6835891924519929
    }
}

# Now use this calibrated depth for more accurate transformations
robot_coords_red_circle = pixel_to_robot_base(168, 353, 0.5208, camera_intrinsics, camera_extrinsics)
robot_coords_red_square = pixel_to_robot_base(364, 436, 0.538-0.009*5, camera_intrinsics, camera_extrinsics)
robot_coords_blue_triangle = pixel_to_robot_base(260, 324, 0.538, camera_intrinsics, camera_extrinsics)
robot_coords_blue_square = pixel_to_robot_base(426, 323, 0.538, camera_intrinsics, camera_extrinsics)

print(f"Calibrated coordinates for pixel (168, 353) red circle: {robot_coords_red_circle}")
print(f"Calibrated coordinates for pixel (260, 324) blue triangle: {robot_coords_blue_triangle}")
print(f"Calibrated coordinates for pixel (342, 324) red square: {robot_coords_red_square}")
print(f"Calibrated coordinates for pixel (426, 323) blue square: {robot_coords_blue_square}")

