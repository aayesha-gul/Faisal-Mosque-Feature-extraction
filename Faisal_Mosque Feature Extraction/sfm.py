import cv2 as cv
import numpy as np
from scipy.optimize import least_squares

def estimate_pose(kp1, kp2, matches, K):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv.findEssentialMat(pts1, pts2, K, method=cv.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv.recoverPose(E, pts1, pts2, K)

    return R, t, pts1, pts2

def triangulate_points(P1, P2, pts1, pts2):
    pts4D = cv.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D = pts4D[:3] / pts4D[3]
    return pts3D
def bundle_adjustment(pts3D, points_2d, camera_params, point_indices, camera_indices, K):
    
    def project_point(params, K, point_3d):
        
        R_vec = params[:3]
        t = params[3:6]
        R, _ = cv.Rodrigues(R_vec)
        point_proj = R @ point_3d + t
        point_proj = point_proj / point_proj[2]  # Normalize
        point_proj = K @ point_proj  # Apply intrinsics
        return point_proj[:2]

    def residuals(params, n_cameras, n_points, points_2d, point_indices, camera_indices, K):
        """Compute residuals between observed and projected points"""
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        
        residuals = []
        for i, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
            
            cam_params = camera_params[cam_idx]
            point_3d = points_3d[pt_idx]
            
            
            point_proj = project_point(cam_params, K, point_3d)
            
            
            residual = points_2d[i] - point_proj
            residuals.extend(residual)
        
        return np.array(residuals)

    
    n_cameras = len(camera_params)
    n_points = pts3D.shape[1]
    
    
    cam_params_flat = []
    for cam_param in camera_params:
        
        R_vec, _ = cv.Rodrigues(cam_param[:, :3])
        t = cam_param[:, 3]
       
        cam_params_flat.append(R_vec[0])
        cam_params_flat.append(R_vec[1])
        cam_params_flat.append(R_vec[2])
        cam_params_flat.append(t[0])
        cam_params_flat.append(t[1])
        cam_params_flat.append(t[2])
    
    
    cam_params_flat = np.array(cam_params_flat)
    
    
    x0 = np.hstack([cam_params_flat, pts3D.T.ravel()])
    
    
    res = least_squares(residuals, x0, args=(n_cameras, n_points, points_2d, 
                                            point_indices, camera_indices, K),
                       verbose=2, ftol=1e-4, method='lm')
    
    
    refined_params = res.x
    refined_cameras = refined_params[:n_cameras * 6].reshape((n_cameras, 6))
    refined_points = refined_params[n_cameras * 6:].reshape((n_points, 3))
    
    return refined_points.T, refined_cameras

