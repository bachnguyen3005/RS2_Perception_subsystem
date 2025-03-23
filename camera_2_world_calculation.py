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
    point_robot[0] = (point_robot [0]+0.03)*1000
    point_robot[1] = (point_robot [1] + 1.06)*1000
    return point_robot

def display_robot_base_coordinates(image, obj, camera_intrinsics, camera_extrinsics, depth=0.5478869-0.09):
    """
    Display the coordinates of an object in the robot base frame.
    
    Parameters:
    - image: The input image
    - obj: The detected object with center coordinates
    - camera_intrinsics: Dictionary with camera intrinsic parameters
    - camera_extrinsics: Dictionary with camera extrinsic parameters
    - depth: Depth value in meters (default 0.5m if no depth camera available)
    """
    if obj is None:
        return
    
    # Get pixel coordinates of the center
    pixel_x, pixel_y = obj['cx'], obj['cy']
    
    # Convert to robot base frame
    robot_coords = pixel_to_robot_base(
        pixel_x, pixel_y, depth, camera_intrinsics, camera_extrinsics
    )
    
    # Draw a marker at the center
    cv2.drawMarker(image, (pixel_x, pixel_y), (0, 255, 255), cv2.MARKER_CROSS, 10, 2)
    
    # Display pixel coordinates
    pixel_text = f"Pixel: ({pixel_x}, {pixel_y})"
    cv2.putText(image, pixel_text, (pixel_x + 10, pixel_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Display robot base coordinates
    x_robot, y_robot, z_robot = robot_coords
    robot_text = f"Robot base: ({x_robot:.3f}, {y_robot:.3f}, {z_robot:.3f})m"
    
    if obj['color'] == "Red" and obj['shape'] == "square":
        y_pos = 150  # Below other status messages
        cv2.putText(image, robot_text, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# Example camera parameters (based on provided data)
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
        'x': -0.03566,
        'y': -0.83704,
        'z': 0.547886
    },
    'rotation': {
        'x': -0.23674,
        'y': 0.230630,
        'z': 0.676611,
        'w': 0.676611
    }
}

# Calibration function to determine optimal depth value
def calibrate_depth(known_pixel_coords, known_robot_coords, camera_intrinsics, camera_extrinsics):
    """
    Calibrate the depth value by finding the depth that minimizes error between
    known robot coordinates and calculated coordinates.
    
    Parameters:
    - known_pixel_coords: (x, y) tuple of pixel coordinates
    - known_robot_coords: (x, y, z) tuple of known robot base coordinates
    - camera_intrinsics: Dictionary with camera intrinsic parameters
    - camera_extrinsics: Dictionary with camera extrinsic parameters
    
    Returns:
    - Optimal depth value (in meters)
    """
    pixel_x, pixel_y = known_pixel_coords
    target_x, target_y, target_z = known_robot_coords
    
    # Try different depth values
    min_error = float('inf')
    optimal_depth = 0.538  # Default depth
    
    for depth in np.linspace(0.1, 2.0, 100):  # Try depths from 0.1m to 2.0m
        coords = pixel_to_robot_base(pixel_x, pixel_y, depth, camera_intrinsics, camera_extrinsics)
        error = np.sqrt((coords[0] - target_x)**2 + (coords[1] - target_y)**2 + (coords[2] - target_z)**2)
        
        if error < min_error:
            min_error = error
            optimal_depth = depth
    
    print(f"Calibrated depth: {optimal_depth:.4f}m (Error: {min_error:.4f}m)")
    return optimal_depth

# Example usage of calibration
# For red square at pixel (342, 324) and known to be at (0, -0.3, 0) in robot base frame
calibrated_depth = calibrate_depth(
    (342, 324),           # Pixel coordinates
    (-0.2, -0.3, 0),         # Known robot base coordinates (meters)
    camera_intrinsics,
    camera_extrinsics
)

# Now use this calibrated depth for more accurate transformations
robot_coords_red_circle = pixel_to_robot_base(168, 353, 0.538, camera_intrinsics, camera_extrinsics)
robot_coords_circle_pose2 = pixel_to_robot_base(338, 351, 0.538, camera_intrinsics, camera_extrinsics)
robot_coords_square = pixel_to_robot_base(342, 324, 0.538, camera_intrinsics, camera_extrinsics)
robot_coords_triangle = pixel_to_robot_base(260, 324, 0.538, camera_intrinsics, camera_extrinsics)
print(f"Calibrated coordinates for pixel (168, 353) red circle: {robot_coords_red_circle}")
print(f"Calibrated coordinates for pixel (338, 351) red circle pose2: {robot_coords_circle_pose2}")
print(f"Calibrated coordinates for pixel (342, 324) red square: {robot_coords_square}")
print(f"Calibrated coordinates for pixel (260, 324) blue triangle: {robot_coords_triangle}")