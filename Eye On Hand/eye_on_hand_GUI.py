import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.spatial.transform import Rotation as R
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

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
    
    return point_base, point_camera[:3], point_ee

class DepthPixelExtractor:
    def __init__(self, ui_callback=None):
        self.bridge = CvBridge()
        self.depth_value = None
        self.ui_callback = ui_callback
        self.latest_depth_image = None
        
        # Subscribe to the depth image topic
        # The topic name may vary depending on your RealSense configuration
        self.depth_topic = '/camera/depth/image_rect_raw'
        self.depth_sub = None  # Will be initialized when needed
        
    def start_subscriber(self):
        """Start the ROS subscriber if not already running"""
        if self.depth_sub is None:
            try:
                self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self.depth_callback)
                return True
            except Exception as e:
                print(f"Error starting ROS subscriber: {e}")
                return False
        return True
        
    def stop_subscriber(self):
        """Stop the ROS subscriber if running"""
        if self.depth_sub is not None:
            self.depth_sub.unregister()
            self.depth_sub = None
    
    def set_topic(self, topic_name):
        """Set a new topic name and restart subscriber"""
        self.depth_topic = topic_name
        self.stop_subscriber()
        return self.start_subscriber()

    def depth_callback(self, depth_msg):
        """Store the latest depth image"""
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            print(f"Error processing depth image: {e}")
    
    def get_depth(self, pixel_x, pixel_y):
        """Extract depth at specified pixel from most recent image"""
        if self.latest_depth_image is None:
            return None
        
        try:
            # Check if coordinates are within image bounds
            height, width = self.latest_depth_image.shape
            if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
                return None
                
            # Extract depth value at the specified pixel coordinates
            depth_value = self.latest_depth_image[pixel_y, pixel_x]
            
            # Convert depth value to meters (depends on your camera and depth encoding)
            # For 16-bit depth images, typical conversion is:
            depth_meters = depth_value * 0.001  # Convert from mm to meters
            
            print(f"Depth at pixel ({pixel_x}, {pixel_y}): {depth_value} units, {depth_meters:.3f} meters")
            
            if self.ui_callback:
                self.ui_callback(depth_meters)
                
            return depth_meters
            
        except Exception as e:
            print(f"Error extracting depth: {e}")
            return None

class CoordinateCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye-on-Hand Camera to World Coordinate Calculator")
        self.root.geometry("800x700")
        
        # Initialize ROS node if needed
        self.ros_initialized = False
        self.depth_extractor = None
        
        self.setup_ui()
        
        # Default camera intrinsics
        self.camera_intrinsics = {
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
        
        # Initialize history list for tracking calculated points
        self.point_history = []
        
    def initialize_ros(self):
        """Initialize ROS node and depth extractor"""
        try:
            if not self.ros_initialized:
                rospy.init_node('coordinate_calculator', anonymous=True, disable_signals=True)
                self.ros_initialized = True
            
            self.depth_extractor = DepthPixelExtractor(ui_callback=self.update_depth_field)
            success = self.depth_extractor.start_subscriber()
            
            if success:
                messagebox.showinfo("ROS Initialization", "ROS node initialized and depth subscriber started.")
                self.ros_status.config(text="Initialized")
            else:
                messagebox.showerror("ROS Error", "Failed to start depth subscriber.")
            
            return success
        except Exception as e:
            messagebox.showerror("ROS Error", f"Failed to initialize ROS: {str(e)}")
            return False
            
    def update_depth_field(self, depth_value):
        """Callback to update depth field with value from ROS"""
        if depth_value is not None:
            self.depth.delete(0, tk.END)
            self.depth.insert(0, f"{depth_value:.6f}")
        
    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        config_tab = ttk.Frame(notebook)
        results_tab = ttk.Frame(notebook)
        ros_tab = ttk.Frame(notebook)
        
        notebook.add(config_tab, text="Configuration")
        notebook.add(results_tab, text="Results")
        notebook.add(ros_tab, text="ROS Settings")
        
        # Configure Configuration Tab
        self.setup_config_tab(config_tab)
        
        # Configure Results Tab
        self.setup_results_tab(results_tab)
        
        # Configure ROS Tab
        self.setup_ros_tab(ros_tab)
        
    def setup_config_tab(self, parent):
        # Camera-to-End Effector Section
        camera_ee_frame = ttk.LabelFrame(parent, text="Camera to End-Effector Transform")
        camera_ee_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(camera_ee_frame, text="Translation (meters)").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Translation inputs (camera to EE)
        self.cam_ee_tx = ttk.Entry(camera_ee_frame, width=10)
        self.cam_ee_tx.grid(row=0, column=1, padx=2, pady=2)
        self.cam_ee_tx.insert(0, "-0.01924")
        
        self.cam_ee_ty = ttk.Entry(camera_ee_frame, width=10)
        self.cam_ee_ty.grid(row=0, column=2, padx=2, pady=2)
        self.cam_ee_ty.insert(0, "-0.05115")
        
        self.cam_ee_tz = ttk.Entry(camera_ee_frame, width=10)
        self.cam_ee_tz.grid(row=0, column=3, padx=2, pady=2)
        self.cam_ee_tz.insert(0, "0.0155")
        
        # Radio buttons for rotation type
        self.cam_ee_rot_type = tk.StringVar(value="quaternion")
        ttk.Label(camera_ee_frame, text="Rotation type:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(camera_ee_frame, text="Quaternion", variable=self.cam_ee_rot_type, value="quaternion", 
                       command=self.toggle_camera_ee_rotation).grid(row=1, column=1, padx=2, pady=2)
        ttk.Radiobutton(camera_ee_frame, text="Euler (RPY)", variable=self.cam_ee_rot_type, value="euler", 
                       command=self.toggle_camera_ee_rotation).grid(row=1, column=2, padx=2, pady=2)
        
        # Quaternion inputs (camera to EE)
        self.cam_ee_quat_frame = ttk.Frame(camera_ee_frame)
        self.cam_ee_quat_frame.grid(row=2, column=0, columnspan=4, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(self.cam_ee_quat_frame, text="Quaternion [x, y, z, w]:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.cam_ee_qx = ttk.Entry(self.cam_ee_quat_frame, width=10)
        self.cam_ee_qx.grid(row=0, column=1, padx=2, pady=2)
        self.cam_ee_qx.insert(0, "0.49578")
        
        self.cam_ee_qy = ttk.Entry(self.cam_ee_quat_frame, width=10)
        self.cam_ee_qy.grid(row=0, column=2, padx=2, pady=2)
        self.cam_ee_qy.insert(0, "-0.50")
        
        self.cam_ee_qz = ttk.Entry(self.cam_ee_quat_frame, width=10)
        self.cam_ee_qz.grid(row=0, column=3, padx=2, pady=2)
        self.cam_ee_qz.insert(0, "0.501087")
        
        self.cam_ee_qw = ttk.Entry(self.cam_ee_quat_frame, width=10)
        self.cam_ee_qw.grid(row=0, column=4, padx=2, pady=2)
        self.cam_ee_qw.insert(0, "0.50245")
        
        # Euler angle inputs (camera to EE) - initially hidden
        self.cam_ee_euler_frame = ttk.Frame(camera_ee_frame)
        self.cam_ee_euler_frame.grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(self.cam_ee_euler_frame, text="Euler angles [roll, pitch, yaw] (radians):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.cam_ee_roll = ttk.Entry(self.cam_ee_euler_frame, width=10)
        self.cam_ee_roll.grid(row=0, column=1, padx=2, pady=2)
        self.cam_ee_roll.insert(0, "0.0")
        
        self.cam_ee_pitch = ttk.Entry(self.cam_ee_euler_frame, width=10)
        self.cam_ee_pitch.grid(row=0, column=2, padx=2, pady=2)
        self.cam_ee_pitch.insert(0, "0.0")
        
        self.cam_ee_yaw = ttk.Entry(self.cam_ee_euler_frame, width=10)
        self.cam_ee_yaw.grid(row=0, column=3, padx=2, pady=2)
        self.cam_ee_yaw.insert(0, "0.0")
        
        # Initially hide Euler inputs for camera to EE
        self.cam_ee_euler_frame.grid_remove()
        
        # End-Effector to Robot Base Section
        ee_base_frame = ttk.LabelFrame(parent, text="End-Effector to Robot Base Transform")
        ee_base_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(ee_base_frame, text="Translation (meters)").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Translation inputs (EE to base)
        self.ee_base_tx = ttk.Entry(ee_base_frame, width=10)
        self.ee_base_tx.grid(row=0, column=1, padx=2, pady=2)
        self.ee_base_tx.insert(0, "-0.13197")
        
        self.ee_base_ty = ttk.Entry(ee_base_frame, width=10)
        self.ee_base_ty.grid(row=0, column=2, padx=2, pady=2)
        self.ee_base_ty.insert(0, "-0.29813")
        
        self.ee_base_tz = ttk.Entry(ee_base_frame, width=10)
        self.ee_base_tz.grid(row=0, column=3, padx=2, pady=2)
        self.ee_base_tz.insert(0, "-0.12682")
        
        # Radio buttons for rotation type
        self.ee_base_rot_type = tk.StringVar(value="euler")
        ttk.Label(ee_base_frame, text="Rotation type:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Radiobutton(ee_base_frame, text="Quaternion", variable=self.ee_base_rot_type, value="quaternion", 
                       command=self.toggle_ee_base_rotation).grid(row=1, column=1, padx=2, pady=2)
        ttk.Radiobutton(ee_base_frame, text="Euler (RPY)", variable=self.ee_base_rot_type, value="euler", 
                       command=self.toggle_ee_base_rotation).grid(row=1, column=2, padx=2, pady=2)
        
        # Quaternion inputs (EE to base) - initially hidden
        self.ee_base_quat_frame = ttk.Frame(ee_base_frame)
        self.ee_base_quat_frame.grid(row=2, column=0, columnspan=4, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(self.ee_base_quat_frame, text="Quaternion [x, y, z, w]:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.ee_base_qx = ttk.Entry(self.ee_base_quat_frame, width=10)
        self.ee_base_qx.grid(row=0, column=1, padx=2, pady=2)
        self.ee_base_qx.insert(0, "0.0")
        
        self.ee_base_qy = ttk.Entry(self.ee_base_quat_frame, width=10)
        self.ee_base_qy.grid(row=0, column=2, padx=2, pady=2)
        self.ee_base_qy.insert(0, "1.0")
        
        self.ee_base_qz = ttk.Entry(self.ee_base_quat_frame, width=10)
        self.ee_base_qz.grid(row=0, column=3, padx=2, pady=2)
        self.ee_base_qz.insert(0, "0.0")
        
        self.ee_base_qw = ttk.Entry(self.ee_base_quat_frame, width=10)
        self.ee_base_qw.grid(row=0, column=4, padx=2, pady=2)
        self.ee_base_qw.insert(0, "0.0")
        
        # Euler angle inputs (EE to base)
        self.ee_base_euler_frame = ttk.Frame(ee_base_frame)
        self.ee_base_euler_frame.grid(row=3, column=0, columnspan=4, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(self.ee_base_euler_frame, text="Euler angles [roll, pitch, yaw] (radians):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        
        self.ee_base_roll = ttk.Entry(self.ee_base_euler_frame, width=10)
        self.ee_base_roll.grid(row=0, column=1, padx=2, pady=2)
        self.ee_base_roll.insert(0, "0.0")
        
        self.ee_base_pitch = ttk.Entry(self.ee_base_euler_frame, width=10)
        self.ee_base_pitch.grid(row=0, column=2, padx=2, pady=2)
        self.ee_base_pitch.insert(0, "3.14")
        
        self.ee_base_yaw = ttk.Entry(self.ee_base_euler_frame, width=10)
        self.ee_base_yaw.grid(row=0, column=3, padx=2, pady=2)
        self.ee_base_yaw.insert(0, "0.0")
        
        # Initially hide quaternion inputs for EE to base
        self.ee_base_quat_frame.grid_remove()
        
        # Pixel and Depth Section
        pixel_frame = ttk.LabelFrame(parent, text="Pixel Coordinates and Depth")
        pixel_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(pixel_frame, text="Pixel X:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.pixel_x = ttk.Entry(pixel_frame, width=10)
        self.pixel_x.grid(row=0, column=1, padx=2, pady=2)
        self.pixel_x.insert(0, "320")
        
        ttk.Label(pixel_frame, text="Pixel Y:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.pixel_y = ttk.Entry(pixel_frame, width=10)
        self.pixel_y.grid(row=0, column=3, padx=2, pady=2)
        self.pixel_y.insert(0, "240")
        
        ttk.Label(pixel_frame, text="Depth (meters):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.depth = ttk.Entry(pixel_frame, width=10)
        self.depth.grid(row=1, column=1, padx=2, pady=2)
        self.depth.insert(0, "0.102")  # 0.126-0.015-0.009 from example
        
        # Get Depth from ROS Button
        get_depth_button = ttk.Button(pixel_frame, text="Get Depth from ROS", command=self.get_depth_from_ros)
        get_depth_button.grid(row=1, column=2, columnspan=2, padx=5, pady=2, sticky=tk.W+tk.E)
        
        # Calculate Button
        calculate_button = ttk.Button(parent, text="Calculate Coordinates", command=self.calculate_coordinates)
        calculate_button.pack(fill=tk.X, padx=5, pady=10)
        
        # Camera intrinsics button
        intrinsics_button = ttk.Button(parent, text="Edit Camera Intrinsics", command=self.edit_camera_intrinsics)
        intrinsics_button.pack(fill=tk.X, padx=5, pady=5)
        
    def setup_results_tab(self, parent):
        # Results Section
        results_frame = ttk.Frame(parent)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(results_frame, text="Camera Frame (x, y, z):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.camera_coords = ttk.Label(results_frame, text="")
        self.camera_coords.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(results_frame, text="End-Effector Frame (x, y, z):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.ee_coords = ttk.Label(results_frame, text="")
        self.ee_coords.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        ttk.Label(results_frame, text="Robot Base Frame (x, y, z):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.base_coords = ttk.Label(results_frame, text="")
        self.base_coords.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # History list
        ttk.Label(results_frame, text="Calculation History:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        
        # Create a frame for the history list with scrollbar
        history_container = ttk.Frame(results_frame)
        history_container.grid(row=4, column=0, columnspan=2, sticky=tk.NSEW, padx=5, pady=2)
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(4, weight=1)
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(history_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create listbox for history
        self.history_listbox = tk.Listbox(history_container, height=15, width=50)
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Connect scrollbar to listbox
        self.history_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.history_listbox.yview)
        
        # Clear history button
        clear_button = ttk.Button(results_frame, text="Clear History", command=self.clear_history)
        clear_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W+tk.E)
        
    def setup_ros_tab(self, parent):
        ros_frame = ttk.Frame(parent)
        ros_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ROS Node Status
        status_frame = ttk.LabelFrame(ros_frame, text="ROS Status")
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(status_frame, text="Node Status:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.ros_status = ttk.Label(status_frame, text="Not Initialized")
        self.ros_status.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Initialize ROS Button
        init_ros_button = ttk.Button(status_frame, text="Initialize ROS", command=self.initialize_ros)
        init_ros_button.grid(row=0, column=2, padx=5, pady=2)
        
        # Topic Settings
        topic_frame = ttk.LabelFrame(ros_frame, text="Depth Topic Settings")
        topic_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(topic_frame, text="Depth Topic:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.depth_topic = ttk.Entry(topic_frame, width=30)
        self.depth_topic.grid(row=0, column=1, padx=2, pady=2)
        self.depth_topic.insert(0, "/camera/depth/image_rect_raw")
        
        set_topic_button = ttk.Button(topic_frame, text="Set Topic", command=self.set_depth_topic)
        set_topic_button.grid(row=0, column=2, padx=5, pady=2)
        
    def toggle_camera_ee_rotation(self):
        if self.cam_ee_rot_type.get() == "quaternion":
            self.cam_ee_quat_frame.grid()
            self.cam_ee_euler_frame.grid_remove()
        else:
            self.cam_ee_quat_frame.grid_remove()
            self.cam_ee_euler_frame.grid()
            
    def toggle_ee_base_rotation(self):
        if self.ee_base_rot_type.get() == "quaternion":
            self.ee_base_quat_frame.grid()
            self.ee_base_euler_frame.grid_remove()
        else:
            self.ee_base_quat_frame.grid_remove()
            self.ee_base_euler_frame.grid()
            
    def edit_camera_intrinsics(self):
        # Create a new window
        intrinsics_window = tk.Toplevel(self.root)
        intrinsics_window.title("Edit Camera Intrinsics")
        intrinsics_window.geometry("500x400")
        
        # Create a frame
        frame = ttk.Frame(intrinsics_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Camera intrinsics components
        ttk.Label(frame, text="fx:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        fx_entry = ttk.Entry(frame, width=15)
        fx_entry.grid(row=0, column=1, padx=2, pady=2)
        fx_entry.insert(0, str(self.camera_intrinsics['K'][0]))
        
        ttk.Label(frame, text="fy:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        fy_entry = ttk.Entry(frame, width=15)
        fy_entry.grid(row=1, column=1, padx=2, pady=2)
        fy_entry.insert(0, str(self.camera_intrinsics['K'][4]))
        
        ttk.Label(frame, text="cx:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        cx_entry = ttk.Entry(frame, width=15)
        cx_entry.grid(row=2, column=1, padx=2, pady=2)
        cx_entry.insert(0, str(self.camera_intrinsics['K'][2]))
        
        ttk.Label(frame, text="cy:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        cy_entry = ttk.Entry(frame, width=15)
        cy_entry.grid(row=3, column=1, padx=2, pady=2)
        cy_entry.insert(0, str(self.camera_intrinsics['K'][5]))
        
        ttk.Label(frame, text="width:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        width_entry = ttk.Entry(frame, width=15)
        width_entry.grid(row=4, column=1, padx=2, pady=2)
        width_entry.insert(0, str(self.camera_intrinsics['width']))
        
        ttk.Label(frame, text="height:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        height_entry = ttk.Entry(frame, width=15)
        height_entry.grid(row=5, column=1, padx=2, pady=2)
        height_entry.insert(0, str(self.camera_intrinsics['height']))
        
        # Save button
        def save_intrinsics():
            try:
                fx = float(fx_entry.get())
                fy = float(fy_entry.get())
                cx = float(cx_entry.get())
                cy = float(cy_entry.get())
                width = int(width_entry.get())
                height = int(height_entry.get())
                
                # Update camera intrinsics
                self.camera_intrinsics['K'][0] = fx
                self.camera_intrinsics['K'][4] = fy
                self.camera_intrinsics['K'][2] = cx
                self.camera_intrinsics['K'][5] = cy
                self.camera_intrinsics['width'] = width
                self.camera_intrinsics['height'] = height
                
                # Also update P matrix
                self.camera_intrinsics['P'][0] = fx
                self.camera_intrinsics['P'][5] = fy
                self.camera_intrinsics['P'][2] = cx
                self.camera_intrinsics['P'][6] = cy
                
                intrinsics_window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers for all fields.")
        
        save_button = ttk.Button(frame, text="Save", command=save_intrinsics)
        save_button.grid(row=6, column=0, columnspan=2, padx=5, pady=10)
    
    def set_depth_topic(self):
        """Set the depth topic for the ROS subscriber"""
        if self.depth_extractor is None:
            messagebox.showerror("Error", "ROS not initialized. Please initialize ROS first.")
            return
        
        new_topic = self.depth_topic.get().strip()
        if not new_topic:
            messagebox.showerror("Error", "Please enter a valid topic name.")
            return
        
        success = self.depth_extractor.set_topic(new_topic)
        if success:
            messagebox.showinfo("Topic Changed", f"Successfully subscribed to topic: {new_topic}")
        else:
            messagebox.showerror("Error", f"Failed to subscribe to topic: {new_topic}")
    
    def get_depth_from_ros(self):
        """Get depth value from ROS at current pixel coordinates"""
        if self.depth_extractor is None:
            if not self.initialize_ros():
                return
        
        try:
            pixel_x = int(float(self.pixel_x.get()))
            pixel_y = int(float(self.pixel_y.get()))
            
            if self.depth_extractor.latest_depth_image is None:
                messagebox.showinfo("Waiting for Data", "No depth image received yet. Please wait for data or check your topic settings.")
                return
            
            depth_value = self.depth_extractor.get_depth(pixel_x, pixel_y)
            
            if depth_value is not None:
                self.depth.delete(0, tk.END)
                self.depth.insert(0, f"{depth_value:.6f}")
                messagebox.showinfo("Depth Value", f"Depth at pixel ({pixel_x}, {pixel_y}): {depth_value:.6f} meters")
            else:
                messagebox.showerror("Error", f"Could not get depth value at pixel ({pixel_x}, {pixel_y}). Pixel might be out of bounds or invalid.")
                
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer values for pixel coordinates.")
    
    def calculate_coordinates(self):
        try:
            # Get pixel and depth values
            pixel_x = float(self.pixel_x.get())
            pixel_y = float(self.pixel_y.get())
            depth = float(self.depth.get())
            
            # Get camera to EE transformation
            cam_ee_translation = [
                float(self.cam_ee_tx.get()),
                float(self.cam_ee_ty.get()),
                float(self.cam_ee_tz.get())
            ]
            
            if self.cam_ee_rot_type.get() == "quaternion":
                cam_ee_rotation = [
                    float(self.cam_ee_qx.get()),
                    float(self.cam_ee_qy.get()),
                    float(self.cam_ee_qz.get()),
                    float(self.cam_ee_qw.get())
                ]
                camera_to_ee_transform = create_transformation_matrix_from_quaternion(
                    cam_ee_translation, cam_ee_rotation)
            else:
                cam_ee_rotation = [
                    float(self.cam_ee_roll.get()),
                    float(self.cam_ee_pitch.get()),
                    float(self.cam_ee_yaw.get())
                ]
                camera_to_ee_transform = create_transformation_matrix_from_euler(
                    cam_ee_translation, cam_ee_rotation)
            
            # Get EE to base transformation
            ee_base_translation = [
                float(self.ee_base_tx.get()),
                float(self.ee_base_ty.get()),
                float(self.ee_base_tz.get())
            ]
            
            if self.ee_base_rot_type.get() == "quaternion":
                ee_base_rotation = [
                    float(self.ee_base_qx.get()),
                    float(self.ee_base_qy.get()),
                    float(self.ee_base_qz.get()),
                    float(self.ee_base_qw.get())
                ]
                ee_to_base_transform = create_transformation_matrix_from_quaternion(
                    ee_base_translation, ee_base_rotation)
            else:
                ee_base_rotation = [
                    float(self.ee_base_roll.get()),
                    float(self.ee_base_pitch.get()),
                    float(self.ee_base_yaw.get())
                ]
                ee_to_base_transform = create_transformation_matrix_from_euler(
                    ee_base_translation, ee_base_rotation)
            
            # Calculate the coordinates
            point_base, point_camera, point_ee = pixel_to_robot_base_eye_on_hand(
                pixel_x, pixel_y, depth, self.camera_intrinsics,
                camera_to_ee_transform, ee_to_base_transform)
            
            # Update the result labels
            self.camera_coords.config(text=f"({point_camera[0]:.6f}, {point_camera[1]:.6f}, {point_camera[2]:.6f})")
            self.ee_coords.config(text=f"({point_ee[0]:.6f}, {point_ee[1]:.6f}, {point_ee[2]:.6f})")
            self.base_coords.config(text=f"({point_base[0]:.6f}, {point_base[1]:.6f}, {point_base[2]:.6f})")
            
            # Add to history
            history_entry = f"Pixel: ({pixel_x}, {pixel_y}), Depth: {depth}m → World: ({point_base[0]:.4f}, {point_base[1]:.4f}, {point_base[2]:.4f})"
            self.history_listbox.insert(tk.END, history_entry)
            
            # Save point for visualization
            self.point_history.append({
                'camera': point_camera,
                'ee': point_ee,
                'base': point_base,
                'pixel': (pixel_x, pixel_y),
                'depth': depth
            })
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def clear_history(self):
        # Clear history list
        self.history_listbox.delete(0, tk.END)
        
        # Clear point history
        self.point_history = []

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = CoordinateCalculatorApp(root)
    root.mainloop()