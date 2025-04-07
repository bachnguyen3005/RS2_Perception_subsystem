import numpy as np
import cv2
import glob
import os
import argparse

def calibrate_camera(images_folder, checkerboard_size, square_size=1.0, visualize=False, save_results=True):
    """
    Calibrate camera intrinsic parameters using a set of checkerboard images.
    
    Parameters:
    -----------
    images_folder : str
        Path to folder containing calibration images
    checkerboard_size : tuple
        Number of internal corners on the checkerboard (width, height)
    square_size : float
        Size of checkerboard squares in your preferred unit (default=1.0)
    visualize : bool
        If True, try to display the detected corners on each image (requires GTK)
    save_results : bool
        If True, save calibration results to a file
        
    Returns:
    --------
    ret : float
        RMS re-projection error
    mtx : ndarray
        Camera matrix (intrinsic parameters)
    dist : ndarray
        Distortion coefficients
    rvecs : list
        Rotation vectors for each image
    tvecs : list
        Translation vectors for each image
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (checkerboard_width-1, checkerboard_height-1, 0)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Get list of calibration images
    images = glob.glob(os.path.join(images_folder, '*.jpg')) + \
             glob.glob(os.path.join(images_folder, '*.png'))
    
    if not images:
        raise ValueError(f"No images found in {images_folder}")
    
    print(f"Found {len(images)} images for calibration")

    # Process each image
    successful_images = 0
    for fname in images:
        print(f"Processing image: {os.path.basename(fname)}")
        img = cv2.imread(fname)
        if img is None:
            print(f"  ERROR: Could not read image file: {fname}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the checkerboard corners - try different flags to improve detection
        print(f"  Looking for {checkerboard_size[0]}×{checkerboard_size[1]} corners...")
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, flags)
        
        if ret:
            print(f"  SUCCESS: Found all {checkerboard_size[0]}×{checkerboard_size[1]} corners")
            
            # If found, add object points and image points
            successful_images += 1
            objpoints.append(objp)
            
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and save the corners image if requested
            if visualize:
                try:
                    img = cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
                    output_dir = 'calibration_corners'
                    os.makedirs(output_dir, exist_ok=True)
                    output_filename = os.path.join(output_dir, f'corners_{os.path.basename(fname)}')
                    cv2.imwrite(output_filename, img)
                    print(f"  Saved corner detection image to {output_filename}")
                except Exception as e:
                    print(f"  Warning: Could not save corner image: {e}")
        else:
            print(f"  FAILED: Could not find checkerboard corners")
            # Save failed images with a prefix for debugging
            debug_dir = "failed_detections"
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, os.path.basename(fname))
            cv2.imwrite(debug_path, img)
            print(f"  Saved failed image to {debug_path}")
            continue
            successful_images += 1
            objpoints.append(objp)
            
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and save the corners image instead of displaying
            if visualize:
                try:
                    img = cv2.drawChessboardCorners(img, checkerboard_size, corners2, ret)
                    output_filename = os.path.join('calibration_corners', f'corners_{os.path.basename(fname)}')
                    os.makedirs('calibration_corners', exist_ok=True)
                    cv2.imwrite(output_filename, img)
                    print(f"Saved corner detection image to {output_filename}")
                except Exception as e:
                    print(f"Warning: Could not save corner image: {e}")
    
    print(f"Successfully processed {successful_images} out of {len(images)} images")
    
    if successful_images < 3:
        raise ValueError("At least 3 successful images are required for calibration")
    
    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    print("\nCamera calibration results:")
    print(f"RMS re-projection error: {ret}")
    print(f"\nCamera matrix:\n{mtx}")
    print(f"\nDistortion coefficients: {dist.ravel()}")
    
    # Calculate re-projection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print(f"\nTotal re-projection error: {mean_error/len(objpoints)}")
    
    # Save calibration results if requested
    if save_results:
        results_file = 'camera_calibration_results.npz'
        np.savez(results_file, 
                 camera_matrix=mtx, 
                 dist_coeffs=dist, 
                 rvecs=rvecs, 
                 tvecs=tvecs, 
                 error=ret)
        print(f"\nCalibration results saved to {results_file}")
    
    return ret, mtx, dist, rvecs, tvecs

def undistort_image(image_path, camera_matrix, dist_coeffs, output_path=None):
    """
    Undistort an image using calibration parameters.
    
    Parameters:
    -----------
    image_path : str
        Path to the image to undistort
    camera_matrix : ndarray
        Camera matrix (intrinsic parameters)
    dist_coeffs : ndarray
        Distortion coefficients
    output_path : str
        Path to save the undistorted image (required)
    """
    if output_path is None:
        output_path = 'undistorted_' + os.path.basename(image_path)
        
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    # Refine camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    
    # Undistort image
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    
    # Crop the image (optional)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    # Save original and undistorted images side by side
    comparison = np.hstack((img, dst))
    comparison_path = 'comparison_' + os.path.basename(output_path)
    cv2.imwrite(comparison_path, comparison)
    
    # Save undistorted image
    cv2.imwrite(output_path, dst)
    print(f"Undistorted image saved to {output_path}")
    print(f"Side-by-side comparison saved to {comparison_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camera Calibration Tool')
    parser.add_argument('--folder', type=str, required=True, 
                        help='Folder containing calibration images')
    parser.add_argument('--chess_width', type=int, required=True, 
                        help='Number of inner corners along checkerboard width')
    parser.add_argument('--chess_height', type=int, required=True, 
                        help='Number of inner corners along checkerboard height')
    parser.add_argument('--square_size', type=float, default=1.0, 
                        help='Size of checkerboard squares in your preferred unit')
    parser.add_argument('--visualize', action='store_true', 
                        help='Save corner detection images to calibration_corners folder')
    parser.add_argument('--undistort', type=str, default=None, 
                        help='Path to image for undistortion test (optional)')
    parser.add_argument('--output', type=str, default=None, 
                        help='Path to save undistorted image (optional)')
    
    args = parser.parse_args()
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(
        args.folder, 
        (args.chess_width, args.chess_height), 
        args.square_size,
        args.visualize
    )
    
    # Test undistortion if requested
    if args.undistort:
        undistort_image(args.undistort, mtx, dist, args.output)