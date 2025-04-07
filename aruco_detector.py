import cv2
import numpy as np
import argparse

def detect_aruco_markers(image_path, marker_size=None, camera_matrix=None, dist_coeffs=None):
    """
    Detect ArUco markers in an image and estimate pose if marker size and camera parameters are provided.
    
    Args:
        image_path (str): Path to the input image.
        marker_size (float, optional): Size of the marker in meters.
        camera_matrix (numpy.ndarray, optional): 3x3 camera intrinsic matrix.
        dist_coeffs (numpy.ndarray, optional): Distortion coefficients.
        
    Returns:
        image (numpy.ndarray): Image with detected markers highlighted.
        marker_info (list): List of dictionaries containing marker information.
    """
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define the ArUco dictionary
    # Available dictionaries: DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000,
    # DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000, DICT_6X6_50,
    # DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000, DICT_7X7_50, DICT_7X7_100,
    # DICT_7X7_250, DICT_7X7_1000, DICT_ARUCO_ORIGINAL
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # Parameters for detection
    parameters = cv2.aruco.DetectorParameters()
    
    # Create detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Detect markers
    marker_corners, marker_ids, rejected = detector.detectMarkers(gray)
    
    marker_info = []
    
    # If markers are detected
    if marker_ids is not None:
        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)
        
        # Process each marker
        for i, (corner, id_) in enumerate(zip(marker_corners, marker_ids)):
            # Get the center of the marker
            center = np.mean(corner[0], axis=0).astype(int)
            
            # Draw the ID number
            cv2.putText(image, f"ID: {id_[0]}", 
                       (center[0], center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            # Calculate corner positions (for reference)
            corners = corner[0].astype(int)
            
            # Initialize pose data
            rvec, tvec = None, None
            
            # If marker size and camera parameters are provided, estimate pose
            if marker_size is not None and camera_matrix is not None and dist_coeffs is not None:
                # Estimate pose for each marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corner], marker_size, camera_matrix, dist_coeffs
                )
                rvec, tvec = rvecs[0][0], tvecs[0][0]
                
                # Draw the coordinate axes
                cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, marker_size * 0.5)
                
                # Calculate distance from camera to marker (in meters)
                distance = np.linalg.norm(tvec)
                
                # Display the distance
                cv2.putText(image, f"{distance:.2f}m", 
                           (center[0], center[1] + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 0, 255), 2)
            
            # Store marker information
            marker_data = {
                'id': id_[0],
                'center': center.tolist(),
                'corners': corners.tolist()
            }
            
            # Add pose information if available
            if rvec is not None and tvec is not None:
                marker_data.update({
                    'rvec': rvec.tolist(),
                    'tvec': tvec.tolist(),
                    'distance': float(np.linalg.norm(tvec))
                })
                
            marker_info.append(marker_data)
    
    return image, marker_info

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Detect ArUco markers in an image")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--output", "-o", help="Path to save the output image", default="output.jpg")
    parser.add_argument("--marker-size", "-s", type=float, help="Size of the marker in meters (e.g., 0.05 for 5cm)")
    parser.add_argument("--calibration", "-c", help="Path to camera calibration file (JSON or YAML)")
    args = parser.parse_args()
    
    # Camera calibration parameters
    camera_matrix = None
    dist_coeffs = None
    
    # Load camera calibration if provided
    if args.calibration:
        if args.calibration.endswith(('.yaml', '.yml')):
            fs = cv2.FileStorage(args.calibration, cv2.FILE_STORAGE_READ)
            camera_matrix = fs.getNode("camera_matrix").mat()
            dist_coeffs = fs.getNode("distortion_coefficients").mat()
            fs.release()
        elif args.calibration.endswith('.json'):
            import json
            with open(args.calibration, 'r') as f:
                calib_data = json.load(f)
                camera_matrix = np.array(calib_data.get("camera_matrix", []))
                dist_coeffs = np.array(calib_data.get("dist_coeffs", []))
        else:
            print(f"Unsupported calibration file format: {args.calibration}")
            print("Please use YAML (.yaml, .yml) or JSON (.json)")
            return
    
    try:
        # Detect markers
        result_image, detected_markers = detect_aruco_markers(
            args.image_path, args.marker_size, camera_matrix, dist_coeffs
        )
        
        # Save the result
        cv2.imwrite(args.output, result_image)
        
        # Print information about detected markers
        if detected_markers:
            print(f"Detected {len(detected_markers)} ArUco markers:")
            for marker in detected_markers:
                print(f"  Marker ID: {marker['id']}")
                print(f"  Center position (px): {marker['center']}")
                
                # Print pose information if available
                if 'distance' in marker:
                    print(f"  Distance from camera: {marker['distance']:.3f} meters")
                    print(f"  Translation vector: {marker['tvec']}")
                    print(f"  Rotation vector: {marker['rvec']}")
                
                print(f"  Corner positions (px): {marker['corners']}")
        else:
            print("No ArUco markers detected in the image.")
            
        print(f"Result saved to {args.output}")
        
        # Display the result (if running in interactive environment)
        cv2.imshow("Detected ArUco Markers", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")

def create_calibration_file(output_path="camera_calibration.json"):
    """
    Create a sample camera calibration file with default values.
    This is just a helper function to generate a template calibration file.
    Real camera calibration should be done properly using calibration patterns.
    
    Args:
        output_path (str): Path to save the calibration file.
    """
    import json
    
    # Example values for a 640x480 camera
    # These are just placeholders and should be replaced with real calibration values
    calibration_data = {
        "camera_matrix": [
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0]
        ],
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "camera_resolution": [640, 480],
        "calibration_date": "2023-01-01",
        "notes": "This is a template calibration file. Replace with actual calibration values."
    }
    
    with open(output_path, 'w') as f:
        json.dump(calibration_data, f, indent=4)
    
    print(f"Sample calibration file created at: {output_path}")
    print("Note: This contains placeholder values. Replace with your actual camera calibration.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--create-calibration":
        output_path = "camera_calibration.json"
        if len(sys.argv) > 2:
            output_path = sys.argv[2]
        create_calibration_file(output_path)
    else:
        main()