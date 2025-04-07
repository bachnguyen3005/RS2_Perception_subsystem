import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def pixel_to_robot_base(pixel_x, pixel_y, depth, camera_intrinsics, camera_extrinsics):
    # Extract intrinsic parameters
    fx = camera_intrinsics['K'][0]
    fy = camera_intrinsics['K'][4]
    cx = camera_intrinsics['K'][2]
    cy = camera_intrinsics['K'][5]
    
    # Extract distortion coefficients
    distortion_coeffs = np.array(camera_intrinsics['D'])
    
    # Extract camera matrix
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # Undistort the pixel coordinates
    distorted_point = np.array([[pixel_x, pixel_y]], dtype=np.float32)
    undistorted_point = cv2.undistortPoints(
        distorted_point, camera_matrix, distortion_coeffs, None, camera_matrix
    )
    
    # Get undistorted pixel coordinates
    undistorted_x = undistorted_point[0][0][0]
    undistorted_y = undistorted_point[0][0][1]
    
    # Convert undistorted pixel coordinates to normalized camera coordinates
    x_normalized = (undistorted_x - cx) / fx
    y_normalized = (undistorted_y - cy) / fy
    
    # Calculate 3D point in camera frame
    point_camera = np.array([
        x_normalized * depth,
        y_normalized * depth,
        depth
    ])
    
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
    
    # Convert quaternion to rotation matrix
    # scipy expects quaternion as [w, x, y, z], but our quaternion is [x, y, z, w]
    rot = R.from_quat([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    rotation_matrix = rot.as_matrix()
    
    # Transform point from camera frame to robot base frame
    # Note: The original code used rotation_matrix @ point_camera - translation
    # However, the proper way to apply extrinsics is: R * point + t
    # Let's verify the original implementation's intention
    point_robot = rotation_matrix @ point_camera - translation
    
    # Apply the same manual adjustments as in the original code
    point_robot[0] = (point_robot[0])
    point_robot[1] = -(point_robot[1])
    point_robot[2] =  -(point_robot[2]+0.1)
    
    return point_robot

# Camera intrinsics
camera_intrinsics = {
    'height': 480,
    'width': 640,
    'distortion_model': 'plumb_bob',
    'D': [0.06117127,  0.1186219,  -0.00319266, -0.00094209, -0.75616137],
    'K': [599.63069353,   0.,         320, 
          0.0, 598.83260821, 240, 
          0.0, 0.0, 1.0],
    'R': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
}

# Camera extrinsics
camera_extrinsics = { #From the first time
    'translation': {
        'x': -0.03566955956972455,
        'y': -0.8370495511785527,
        'z': 0.5478866489864528
    },
    'rotation': {
        'x': -0.23677471329049302,
        'y': 0.230630756583166,
        'z': 0.6579844217791484,
        'w': 0.6766119197590601
    }
}


def pixel_to_3d_point_cv2(pixel_x, pixel_y, depth, camera_intrinsics, camera_extrinsics):
    """
    Alternative implementation using more OpenCV functions
    """
    # Extract camera matrix and distortion coefficients
    camera_matrix = np.array([
        [camera_intrinsics['K'][0], 0, camera_intrinsics['K'][2]],
        [0, camera_intrinsics['K'][4], camera_intrinsics['K'][5]],
        [0, 0, 1]
    ])
    dist_coeffs = np.array(camera_intrinsics['D'])
    
    # Convert extrinsics to OpenCV format
    quaternion = np.array([
        camera_extrinsics['rotation']['x'],
        camera_extrinsics['rotation']['y'],
        camera_extrinsics['rotation']['z'],
        camera_extrinsics['rotation']['w']
    ])
    
    # Convert quaternion to rotation matrix (scipy expects [w, x, y, z])
    rot = R.from_quat([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
    rvec, _ = cv2.Rodrigues(rot.as_matrix())
    
    tvec = np.array([
        camera_extrinsics['translation']['x'],
        camera_extrinsics['translation']['y'],
        camera_extrinsics['translation']['z']
    ])
    
    # Undistort and normalize the pixel coordinates
    points = np.array([[pixel_x, pixel_y]], dtype=np.float32)
    undistorted_normalized = cv2.undistortPoints(points, camera_matrix, dist_coeffs)
    
    # The undistorted points are now in normalized image coordinates
    # We need to create a ray in camera coordinates
    ray_camera = np.array([undistorted_normalized[0][0][0], 
                          undistorted_normalized[0][0][1], 
                          1.0])
    
    # Scale the ray by depth
    point_camera = ray_camera * depth
    
    # Transform point from camera to world (robot base) coordinates
    rotation_mat, _ = cv2.Rodrigues(rvec)
    point_world = rotation_mat @ point_camera - tvec
    
    # Apply the manual adjustments as in your original code
    point_world[0] = -(point_world[0] )
    point_world[1] = -(point_world[1] )
    
    return point_world

# Verify with your new calibration parameters
updated_camera_intrinsics = {
    'height': 480,
    'width': 640,
    'distortion_model': 'plumb_bob',
    'D': [0.06117127, 0.1186219, -0.00319266, -0.00094209, -0.75616137],
    'K': [596.68849646, 0.0, 320, 
          0.0, 596.0051831, 240, 
          0.0, 0.0, 1.0],
    'R': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    'P': [596.68849646, 0.0, 317.0835119, 0.0,
          0.0, 596.0051831, 247.34658383, 0.0,
          0.0, 0.0, 1.0, 0.0]
}

camera_extrinsics = { 
    'translation': {
        'x': -0.03528364836810119,
        'y': -0.8899924565273498,
        'z': 0.5199620895203437
    },
    'rotation': {
        'x': -0.15756178890452707,
        'y': 0.212698913863979,
        'z': 0.6903460002989902,
        'w': 0.6733170535412542
    }
}

# Calculate coordinates for the same red square now with distortion correction
robot_coords_red_square = pixel_to_robot_base(341, 324, 0.53, camera_intrinsics, camera_extrinsics)
print(f"Recalculated coordinates with distortion correction for pixel (340, 389) red square: {robot_coords_red_square}")




robot_coords_alt = pixel_to_3d_point_cv2(341, 324, 0.53, updated_camera_intrinsics, camera_extrinsics)
print(f"Coordinates using alternative OpenCV method: {robot_coords_alt}")