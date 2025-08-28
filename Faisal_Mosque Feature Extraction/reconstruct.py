import numpy as np
import cv2
from io_utils import load_images, save_ply
from features import extract_features, match_features
from sfm import estimate_pose, triangulate_points, bundle_adjustment

def main():
    
    print("Loading images...")
    images = load_images("faisal_mosque_images")
    print(f"Total images loaded: {len(images)}")

    
    h, w = images[0].shape[:2]
    K = np.array([[1000, 0, w//2],
                  [0, 1000, h//2],
                  [0, 0, 1]])

    
    print("Extracting features...")
    keypoints, descriptors = extract_features(images)

    
    all_points_3d = np.zeros((3, 0))        
    all_points_2d = []                      
    point_indices = []                      
    camera_indices = []                     
    camera_params = []                      

    
    print("Processing images 0 and 1...")
    matches = match_features(descriptors[0], descriptors[1])
    R, t, pts1, pts2 = estimate_pose(keypoints[0], keypoints[1], matches, K)

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))
    P1_proj = K @ P1
    P2_proj = K @ P2

    pts3D = triangulate_points(P1_proj, P2_proj, pts1, pts2)

    
    all_points_3d = pts3D
    
    
    for i in range(pts3D.shape[1]):
        all_points_2d.append(pts1[i])
        point_indices.append(i)
        camera_indices.append(0)
        
        all_points_2d.append(pts2[i])
        point_indices.append(i)
        camera_indices.append(1)

    
    
    camera_params.append(P1)
    camera_params.append(P2)

    
    all_points_2d = np.array(all_points_2d)
    point_indices = np.array(point_indices)
    camera_indices = np.array(camera_indices)

    print(f"Initial reconstruction: {all_points_3d.shape[1]} points")

    
    print("Running bundle adjustment...")
    all_points_3d, refined_cameras = bundle_adjustment(
        all_points_3d, all_points_2d, camera_params, 
        point_indices, camera_indices, K
    )
    
    for i, cam_params in enumerate(refined_cameras):
        R_vec = cam_params[:3]
        t_vec = cam_params[3:]
        R, _ = cv2.Rodrigues(R_vec)
        camera_params[i] = np.hstack((R, t_vec.reshape(3, 1)))
    
    print("Bundle adjustment completed!")

    
    for i in range(2, len(images)):
        print(f"\nProcessing image {i}...")
        
        
        matches = match_features(descriptors[i-1], descriptors[i])
        pts_prev = np.float32([keypoints[i-1][m.queryIdx].pt for m in matches])
        pts_curr = np.float32([keypoints[i][m.trainIdx].pt for m in matches])
        
        
        object_points = []
        image_points = []
        valid_indices = []
        
        for match_idx, pt_curr in enumerate(pts_curr):
            
            
            if match_idx < all_points_3d.shape[1]:
                object_points.append(all_points_3d[:, match_idx])
                image_points.append(pt_curr)
                valid_indices.append(match_idx)
        
        if len(object_points) >= 6:
            object_points = np.array(object_points).reshape(-1, 3)
            image_points = np.array(image_points).reshape(-1, 2)
            
            
            success, R_vec, t_vec, inliers = cv2.solvePnPRansac(
                object_points, image_points, K, None,
                flags=cv2.SOLVEPNP_ITERATIVE, confidence=0.99, reprojectionError=2.0
            )
            
            if success and len(inliers) > 10:
                R_new, _ = cv2.Rodrigues(R_vec)
                t_new = t_vec.reshape(3, 1)
                P_new = np.hstack((R_new, t_new))
                camera_params.append(P_new)
                
                
                for inlier in inliers:
                    idx = valid_indices[inlier[0]]
                    all_points_2d = np.vstack([all_points_2d, image_points[inlier[0]]])
                    point_indices = np.append(point_indices, idx)
                    camera_indices = np.append(camera_indices, i)
                
                print(f"Added {len(inliers)} points to camera {i}")
            
        
        if i % 5 == 0:
            print("Running periodic bundle adjustment...")
            all_points_3d, refined_cameras = bundle_adjustment(
                all_points_3d, all_points_2d, camera_params, 
                point_indices, camera_indices, K
            )
            
            for j, cam_params in enumerate(refined_cameras):
                R_vec = cam_params[:3]
                t_vec = cam_params[3:]
                R, _ = cv2.Rodrigues(R_vec)
                camera_params[j] = np.hstack((R, t_vec.reshape(3, 1)))
            print("Bundle adjustment completed!")


    save_ply("sparse_cloud.ply", all_points_3d)
    print(f"Final reconstruction: {all_points_3d.shape[1]} points")
    print("Reconstruction completed!")

if __name__ == "__main__":
    main()