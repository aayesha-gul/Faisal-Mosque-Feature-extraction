import numpy as np
import cv2
from io_utils import load_images, save_ply
from features import extract_features, match_features
from sfm import estimate_pose, triangulate_points, bundle_adjustment

# Load dataset
images = load_images("faisal_mosque_images")

# Camera intrinsics
h, w = images[0].shape
K = np.array([[1000, 0, w//2],
              [0, 1000, h//2],
              [0, 0, 1]])

# Extract features
keypoints, descriptors = extract_features(images)

# Data structures for bundle adjustment
all_points_3d = []          # List to store all 3D points
all_points_2d = []          # List to store all 2D observations
point_indices = []          # Which 3D point each observation corresponds to
camera_indices = []         # Which camera each observation comes from
camera_params = []          # Store camera parameters [R|t] for each camera

# Start with first two images
print("Processing images 0 and 1...")
matches = match_features(descriptors[0], descriptors[1])
R, t, pts1, pts2 = estimate_pose(keypoints[0], keypoints[1], matches, K)

P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = np.hstack((R, t))
P1_proj = K @ P1
P2_proj = K @ P2

pts3D = triangulate_points(P1_proj, P2_proj, pts1, pts2)

# Store initial points and observations
for i in range(pts3D.shape[1]):
    all_points_3d.append(pts3D[:, i])
    all_points_2d.append(pts1[i])    # Observation from camera 0
    point_indices.append(len(all_points_3d)-1)
    camera_indices.append(0)
    
    all_points_2d.append(pts2[i])    # Observation from camera 1  
    point_indices.append(len(all_points_3d)-1)
    camera_indices.append(1)

# Store camera parameters
camera_params.append(P1)  # Camera 0
camera_params.append(P2)  # Camera 1

# Convert to numpy arrays
all_points_3d = np.array(all_points_3d).T  # Convert to (3, n_points)
all_points_2d = np.array(all_points_2d)
point_indices = np.array(point_indices)
camera_indices = np.array(camera_indices)

print(f"Initial reconstruction: {all_points_3d.shape[1]} points, {len(all_points_2d)} observations")

# Run bundle adjustment after first two cameras
print("Running bundle adjustment...")
refined_points, refined_cameras = bundle_adjustment(
    all_points_3d, all_points_2d, camera_params, 
    point_indices, camera_indices, K
)
all_points_3d = refined_points
camera_params = [np.hstack((cv2.Rodrigues(cam[:3])[0], cam[3:].reshape(3,1))) for cam in refined_cameras]
print("Bundle adjustment completed!")

# Incremental reconstruction for remaining images
for i in range(2, len(images)):
    print(f"\nProcessing image {i}...")
    
    # Find matches with previous image
    matches = match_features(descriptors[i-1], descriptors[i])
    pts_prev = np.float32([keypoints[i-1][m.queryIdx].pt for m in matches])
    pts_curr = np.float32([keypoints[i][m.trainIdx].pt for m in matches])
    
    # Find which points are already reconstructed
    object_points = []
    image_points = []
    valid_matches = []
    
    for match_idx, (pt_prev, pt_curr) in enumerate(zip(pts_prev, pts_curr)):
        # Check if this point from previous image exists in our 3D points
        for obs_idx in range(len(all_points_2d)):
            if (camera_indices[obs_idx] == i-1 and 
                np.allclose(all_points_2d[obs_idx], pt_prev, atol=1.0)):
                point_idx = point_indices[obs_idx]
                object_points.append(all_points_3d[:, point_idx])
                image_points.append(pt_curr)
                valid_matches.append(match_idx)
                break
    
    if len(object_points) >= 6:  # Need at least 6 points for PnP
        object_points = np.array(object_points)
        image_points = np.array(image_points)
        
        # Estimate new camera pose using solvePnP
        success, R_vec, t_vec, inliers = cv2.solvePnPRansac(
            object_points, image_points, K, None,
            flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.99, reprojectionError=2.0
        )
        
        if success:
            R_new, _ = cv2.Rodrigues(R_vec)
            t_new = t_vec.reshape(3, 1)
            P_curr = np.hstack((R_new, t_new))
            camera_params.append(P_curr)
            
            # Add new observations for existing points
            for inlier_idx in range(len(inliers)):
                inlier = inliers[inlier_idx][0]
                point_idx = point_indices[valid_matches[inlier]]
                all_points_2d = np.vstack([all_points_2d, image_points[inlier]])
                point_indices = np.append(point_indices, point_idx)
                camera_indices = np.append(camera_indices, i)
            
            print(f"Added {len(inliers)} existing points to camera {i}")
    
    # Triangulate new points from unmatched features
    # (This part would need additional implementation)
    
    # Run bundle adjustment every 5 cameras
    if i % 5 == 0:
        print("Running periodic bundle adjustment...")
        refined_points, refined_cameras = bundle_adjustment(
            all_points_3d, all_points_2d, camera_params, 
            point_indices, camera_indices, K
        )
        all_points_3d = refined_points
        camera_params = [np.hstack((cv2.Rodrigues(cam[:3])[0], cam[3:].reshape(3,1))) for cam in refined_cameras]
        print("Bundle adjustment completed!")

# Save final sparse cloud
save_ply("sparse_cloud.ply", all_points_3d)
print(f"Final reconstruction: {all_points_3d.shape[1]} points")
print("Reconstruction completed!")