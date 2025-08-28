import cv2 as cv

def extract_features(images):
    orb = cv.ORB_create(5000)
    keypoints, descriptors = [], []
    for idx, img in enumerate(images):
        kp, des = orb.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)
        print(f"Image {idx+1}: {len(kp)} keypoints detected")
    return keypoints, descriptors

def match_features(des1, des2):
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches
def visualize_matches(img1, kp1, img2, kp2, matches, filename="matches.jpg"):
    """Visualize feature matches between two images"""
    
    match_img = cv.drawMatches(img1, kp1, img2, kp2, matches[:50], None, 
                              flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite(filename, match_img)
    print(f"Saved match visualization to {filename}")
    return match_img
