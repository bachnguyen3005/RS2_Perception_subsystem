import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def pixel_to_robot_base(pixel_x, pixel_y, depth, camera_intrinsics, camera_extrinsics):

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
    
    #Convert pixel coordinates to normalized camera coordinates
    x_normalized = (pixel_x - cx) / fx
    y_normalized = (pixel_y - cy) / fy
    
    #Calculate 3D point in camera frame
    point_camera = np.array([
        x_normalized * depth,
        y_normalized * depth,
        depth
    ])
    
    #Convert quaternion to rotation matrix
    #scipy expects quaternion as [w, x, y, z], but our quaternion is [x, y, z, w]
    rot = R.from_quat([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    rotation_matrix = rot.as_matrix()
    
    #Transform point from camera frame to robot base frame
    point_robot = rotation_matrix @ point_camera - translation
    
    
    point_robot[0] = (point_robot [0]+0.02)
    point_robot[1] = -(point_robot [1]+0.1)
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

camera_extrinsics = { #26 of march
    'translation': {
        'x': -0.02949,
        'y': -0.871109,
        'z': 0.51042
    },
    'rotation': {
        'x': -0.184867,
        'y': 0.17938345,
        'z': 0.6766299,
        'w': 0.689795
        
    }
    }
robot_coords_red_square = pixel_to_robot_base (260, 435, 0.576,  camera_intrinsics, camera_extrinsics)
print(f"Calibrated coordinates for pixel (323, 386) red square: {robot_coords_red_square}")


