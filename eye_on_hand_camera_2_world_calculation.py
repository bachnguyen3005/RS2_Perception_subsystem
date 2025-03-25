import numpy as np
from scipy.spatial.transform import Rotation as R

def create_transformation_matrix_from_quaternion(translation, rotation_quat):
    """
    Create a 4x4 homogeneous transformation matrix from translation vector and quaternion.
    
    Parameters:
    - translation: [x, y, z] translation vector
    - rotation_quat: [x, y, z, w] quaternion
    
    Returns:
    - 4x4 transformation matrix
    """
    # Convert quaternion to rotation matrix
    # Note: scipy expects quaternion as [w, x, y, z], but our quaternion is [x, y, z, w]
    rot = R.from_quat([rotation_quat[3], rotation_quat[0], rotation_quat[1], rotation_quat[2]])
    rotation_matrix = rot.as_matrix()
    
    # Create transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    
    return transform

def create_transformation_matrix_from_euler(translation, rotation_rpy):
    """
    Create a 4x4 homogeneous transformation matrix from translation vector and roll-pitch-yaw angles.
    
    Parameters:
    - translation: [x, y, z] translation vector
    - rotation_rpy: [roll, pitch, yaw] in radians
    
    Returns:
    - 4x4 transformation matrix
    """
    # Extract roll, pitch, yaw
    roll, pitch, yaw = rotation_rpy
    
    # Use scipy's Rotation class to create rotation from Euler angles
    # Assuming ZYX convention (yaw, pitch, roll)
    rot = R.from_euler('zyx', [yaw, pitch, roll])
    rotation_matrix = rot.as_matrix()
    
    # Create transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    
    return transform

def pixel_to_camera(pixel_x, pixel_y, depth, camera_intrinsics):
    """
    Convert pixel coordinates to 3D point in camera frame.
    
    Parameters:
    - pixel_x, pixel_y: Pixel coordinates in the image
    - depth: Depth value in meters at the pixel
    - camera_intrinsics: Dictionary with camera intrinsic parameters
    
    Returns:
    - point_camera: 3D point in camera frame
    """
    # Extract intrinsic parameters
    fx = camera_intrinsics['K'][0]
    fy = camera_intrinsics['K'][4]
    cx = camera_intrinsics['K'][2]
    cy = camera_intrinsics['K'][5]
    
    # Convert pixel coordinates to normalized camera coordinates
    x_normalized = (pixel_x - cx) / fx
    y_normalized = (pixel_y - cy) / fy
    
    # Calculate 3D point in camera frame
    point_camera = np.array([
        x_normalized * depth,
        y_normalized * depth,
        depth,
        1.0  # Homogeneous coordinate
    ])
    
    return point_camera

def transform_point(point, transformation_matrix):
    """
    Transform a point using a transformation matrix.
    
    Parameters:
    - point: 4D homogeneous point [x, y, z, 1]
    - transformation_matrix: 4x4 transformation matrix
    
    Returns:
    - transformed_point: Transformed 3D point
    """
    transformed_point = transformation_matrix @ point
    return transformed_point[:3]  # Return only x, y, z

def pixel_to_robot_base_eye_on_hand(pixel_x, pixel_y, depth, camera_intrinsics, 
                                   camera_to_ee_transform, ee_to_base_transform):
    """
    Transform pixel coordinates to robot base frame with eye-on-hand configuration.
    
    Parameters:
    - pixel_x, pixel_y: Pixel coordinates in the image
    - depth: Depth value in meters at the pixel
    - camera_intrinsics: Dictionary with camera intrinsic parameters
    - camera_to_ee_transform: 4x4 transformation matrix from camera to end-effector
    - ee_to_base_transform: 4x4 transformation matrix from end-effector to robot base
    
    Returns:
    - point_base: Coordinates in the robot base frame (in meters)
    """
    # 1. Convert pixel to 3D point in camera frame
    point_camera = pixel_to_camera(pixel_x, pixel_y, depth, camera_intrinsics)
    
    # 2. Transform point from camera frame to end-effector frame
    point_ee = transform_point(point_camera, camera_to_ee_transform)
    
    # 3. Transform point from end-effector frame to robot base frame
    point_base = transform_point(np.append(point_ee, 1.0), ee_to_base_transform)
    
    return point_base

# Example usage:
if __name__ == "__main__":
    # Camera intrinsics (same as in the original script)
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
    
    # Camera-to-end-effector calibration using quaternion (from calibration)
    camera_to_ee_translation = [-0.01924, -0.05115, 0.0155]  # meters
    camera_to_ee_rotation = [0.49578, -0.50, 0.501087, 0.50245]   # quaternion [x, y, z, w]
    camera_to_ee_transform = create_transformation_matrix_from_quaternion(
        camera_to_ee_translation, camera_to_ee_rotation)
    
    # Current end-effector pose using roll-pitch-yaw (from robot controller)
    ee_to_base_translation = [-0.13197, -0.29813, -0.12682]  # meters
    ee_to_base_rotation_rpy = [0.0, 3.14, 0]  # radians [roll, pitch, yaw]
    ee_to_base_transform = create_transformation_matrix_from_euler(
        ee_to_base_translation, ee_to_base_rotation_rpy)
    
    # Example: Calculate object position from a pixel coordinate
    pixel_x, pixel_y = 320, 240  # Center of image
    depth = 0.126-0.015-0.009  # meters
    
    # Calculate object position in robot base frame
    object_position = pixel_to_robot_base_eye_on_hand(
        pixel_x, pixel_y, depth, camera_intrinsics, 
        camera_to_ee_transform, ee_to_base_transform)
    
    print(f"Object position in robot base frame: {object_position} meters")
    
    # Function for calculating multiple object positions
    def calculate_object_positions(pixel_coordinates, depth_values, camera_intrinsics, 
                                  camera_to_ee_transform, ee_pose):
        """
        Calculate positions of multiple objects in robot base frame.
        
        Parameters:
        - pixel_coordinates: List of (x, y) pixel coordinates
        - depth_values: List of corresponding depth values
        - camera_intrinsics: Camera intrinsic parameters
        - camera_to_ee_transform: Transform from camera to end-effector
        - ee_pose: End-effector pose as [position, orientation]
              where position is [x, y, z] and orientation is [roll, pitch, yaw]
        
        Returns:
        - List of object positions in robot base frame
        """
        ee_position, ee_orientation_rpy = ee_pose
        ee_to_base_transform = create_transformation_matrix_from_euler(
            ee_position, ee_orientation_rpy)
        
        object_positions = []
        for (px, py), depth in zip(pixel_coordinates, depth_values):
            pos = pixel_to_robot_base_eye_on_hand(
                px, py, depth, camera_intrinsics, 
                camera_to_ee_transform, ee_to_base_transform)
            object_positions.append(pos)
        
        return object_positions
    
    # Example for calculating multiple object positions
    pixel_coordinates = [
        (249, 404)  # red square

    ]
    depth_values = [0.126-0.015-0.009, 0.538, 0.538, 0.538]
    
    # Current end-effector pose [position, orientation]
    ee_pose = [
        [-0.13197, -0.29813, -0.12682],  # position [x, y, z]
        [0.0, 3.14, 0.0]  # orientation [roll, pitch, yaw]
    ]
    
    # Calculate all object positions
    positions = calculate_object_positions(
        pixel_coordinates, depth_values, camera_intrinsics,
        camera_to_ee_transform, ee_pose)
    
    # Print results
    for i, ((px, py), pos) in enumerate(zip(pixel_coordinates, positions)):
        print(f"Object {i+1} at pixel ({px}, {py}) position: {pos} meters")