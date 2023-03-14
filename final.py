import cv2
import argparse
import numpy as np
import os
from random import choices

def load_images(DIR):
    fn = [os.path.join(DIR, f) for f in sorted(os.listdir(DIR)) if f.endswith(".JPG")]
    return np.array([cv2.imread(f) for f in fn]), np.array(
        [cv2.imread(f, 0) for f in fn]
    )

def nonmax_suppression(img, window_size=5):
    """
    Apply non-maximum suppression to an image
    ## Returns:
        img_copy: a copy of the image with non-maximum suppression applied
    """
    img_copy = img.copy()
    img_min = img.min()

    p = window_size // 2
    for r, c in np.ndindex(img_copy.shape):
        # get window around specific pixel
        c_lower = max(0, c - p)
        c_upper = min(img_copy.shape[1], c + p)
        r_lower = max(0, r - p)
        r_upper = min(img_copy.shape[0], r + p)
        
        # set pixel to img_min so it is not included in max calculation
        temp = img_copy[r, c]
        img_copy[r, c] = img_min

        # if pixel is the max in the window, keep it, otherwise keep it img_min
        if temp > img_copy[r_lower:r_upper, c_lower:c_upper].max():
            img_copy[r, c] = temp
    
    return img_copy


def harris_corner_detector(image, gray, window_size=5, k=0.04, num_corners=500, slice_size=7):
    """
    Detect Harris corners in an image, returning their locations and neighborhoods
    ## Returns:
        corners: (num_corners, 2) array of (x, y) coordinates of the corners
        neighborhood: (num_corners, neighborhood_size, neighborhood_size) array of the neighborhoods around the corners
    """
    # Compute the gradients
    Ix = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0, ksize=window_size)
    Iy = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=window_size)

    # Compute the products of the gradients
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # Compute the sums of the products of the gradients
    S2x = cv2.GaussianBlur(Ix2, (window_size, window_size), 0)
    S2y = cv2.GaussianBlur(Iy2, (window_size, window_size), 0)
    Sxy = cv2.GaussianBlur(Ixy, (window_size, window_size), 0)

    # Harris Corner Response
    R = np.empty(shape=S2x.shape, dtype=np.float32)
    p = slice_size // 2
    for i, j in np.ndindex(S2x.shape):
        # set edges to zero as we cannot give them features easily
        if i < p or i >= (R.shape[0] - p) or j < p or j >= (R.shape[1] - p):
            R[i, j] = 0
            continue
        # calculate R value
        C = np.array([[S2x[i, j], Sxy[i, j]], [Sxy[i, j], S2y[i, j]]])
        R[i, j] = np.linalg.det(C) - k * (np.trace(C) ** 2)

    # Calculate Non-maximum suppression
    Rs = nonmax_suppression(R)

    # Return the top num_corners corners by sorting and returning the indices
    corners = np.unravel_index(np.argsort(Rs, axis=None)[-num_corners:], R.shape)
    corners = np.stack((corners[1], corners[0]), axis=1)

    # Get the neighborhoods of the corners
    neighborhoods = np.empty((num_corners, slice_size, slice_size, image.shape[2]))
    for i, (x, y) in enumerate(corners):
        neighborhoods[i] = image[y - p:y + p + 1, x - p:x + p + 1]

    return corners, neighborhoods


def ncc(corners1, corners2, slices1, slices2, max_correspondences_per_feature=5):
    """
    Find correspondences between the two images, returned as a dictionary mapping the corners
    from image1 to the corners in image2
    
    ## Returns:
        correspondences: dictionary mapping (x1, y1) to (x2, y2)
    """
    correspondences = {}

    # Calculate mean for normalization
    f_bar = slices1.mean(axis=0, keepdims=True)
    g_bar = slices2.mean(axis=0, keepdims=True)
    # Calculate standard deviation for normalization
    f_std = np.linalg.norm(slices1, axis=0)
    g_std = np.linalg.norm(slices2, axis=0)

    found_features = {}
    for corner1, f in zip(corners1, slices1):
        best_ncc = -2
        best_corner = None

        f_hat = (f - f_bar) / f_std

        for corner2, g in zip(corners2, slices2):
            g_hat = (g - g_bar) / g_std
            Nfg = np.sum(f_hat * g_hat)

            if Nfg > best_ncc:
                best_corner = corner2
                best_ncc = Nfg

        if best_ncc < 0:
            continue

        count = found_features.get(tuple(best_corner), 0)
        if count < max_correspondences_per_feature:
            found_features[tuple(best_corner)] = count + 1
            correspondences[tuple(corner1)] = best_corner

    return correspondences


def ransac(correspondences):
    """
    Estimate the homography between the two images using the given correspondences
    ## Returns: 
        homography: 3x3 homography matrix
    """
    # Implement RANSAC for homography estimation
    k = 1000 # Iterations
    N = 4 # Number of samples required
    H = None

    max_inliers = -1
    num_inliers = 0

    inliers = {}
    set_inliers = {}
    
    dist_threshold = 1

    for _ in range(k):
        set_inliers = {}
        num_inliers = 0

        # Sample 4 points from the correspondences
        points1 = choices(list(correspondences.keys()), k=4)
        points2 = [tuple(correspondences.get(p)) for p in points1]

        # Calculate homography
        h, status = cv2.findHomography(np.asarray(points1), np.asarray(points2))

        # Calculate inliers
        for c1, c2 in correspondences.items():
            # Get the points
            pt1 = np.array([c1[0], c1[1], 1])
            pt2 = np.array([c2[0], c2[1], 1])

            # Compute the homography transformation
            pt2_hat = np.matmul(h, pt1.T)
            # Normalize
            pt2_hat /= pt2_hat[2]
            # Calculate distance to expected point
            dist = np.linalg.norm(pt2 - pt2_hat)

            # Check if outlier
            if dist > dist_threshold:
                continue

            # Store inlier
            num_inliers += 1
            set_inliers[c1] = c2

        # Check if the homography is better than the previous one
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            inliers = set_inliers
            H = h

    # Return homography and inliers
    return H, inliers


def warp_image(img1, img2, homography):
    """
    Warp one image onto the other one, blending overlapping pixels together to create
    a single image that shows the union of all pixels from both input images
    ## Returns:
        output: the blended image
    """

    left_img = img1.copy()
    right_img = img2.copy()
    # Check if img2 is to the left of img1
    if homography[0, 2] > 0:
        # Swap the images
        left_img, right_img = right_img, left_img
    else:
        homography = np.linalg.inv(homography)

    result = cv2.warpPerspective(right_img, homography, 
                                   (left_img.shape[1] + right_img.shape[1], left_img.shape[0]))
    
    result[0:left_img.shape[0], 0:left_img.shape[1]] = left_img
    return result


##########################
# Display helper functions
##########################


def display_harris_corners(img1, corners1, img2=None, corners2=None):
    """
    Display the Harris corners on top of the image
    """
    img1_copy = img1.copy()
    for corner in corners1:
        cv2.circle(img1_copy, corner, 2, (0, 0, 255), -1)
    if img2 is not None:
        img2_copy = img2.copy()
        for corner in corners2:
            cv2.circle(img2_copy, corner, 2, (0, 0, 255), -1)
        cv2.imshow("harris corners", np.concatenate((img1_copy, img2_copy), axis=1))
        cv2.imwrite("output_harris_corners.jpg", np.concatenate((img1_copy, img2_copy), axis=1))
    else:
        cv2.imshow("harris corners", img1_copy)
        cv2.imwrite("output_harris_corners.jpg", img1_copy)


def display_correspondences(img1, img2, correspondences, inliers=None):
    """
    Display the correspondences between the two images one on top of the other with lines
    """

    images = np.concatenate((img1, img2), axis=1) 
    for (c1r, c1c), (c2r, c2c) in correspondences.items():
        cv2.circle(images, (c1r, c1c), 2, (255, 0, 0), -1)
        cv2.circle(images, (c2r+img1.shape[1], c2c), 2, (255, 0, 0), -1)
        cv2.line(images, (c1r, c1c), (c2r+img1.shape[1], c2c), thickness=1, color=(0, 0, 255))
    if inliers is not None:
        for (c1r, c1c), (c2r, c2c) in inliers.items():
            cv2.line(images, (c1r, c1c), (c2r+img1.shape[1], c2c), thickness=1, color=(0, 255, 0))
    cv2.imshow("correspondences", images)
    cv2.imwrite("output_correspondences.jpg", images)


def main():
    # Read in Images
    DIR = "DanaHallWay1"  # --> k=0.0025
    # DIR = "DanaOffice" #--> k=0.01 (?)
    col, gray = load_images(DIR)

    image1, image2 = col[0], col[1]
    gray1, gray2 = gray[0], gray[1]

    cv2.imshow("input images", np.concatenate((image1, image2), axis=1))

    # Harris corner detection
    window_size = 5
    corners1, slices1 = harris_corner_detector(image1, gray1, window_size, slice_size=19)
    corners2, slices2 = harris_corner_detector(image2, gray2, window_size, slice_size=19)
    display_harris_corners(image1, corners1, image2, corners2)

    # Compute NCC for correspondences
    correspondences = ncc(corners1, corners2, slices1, slices2)

    # Perform RANSAC to estimate the homography matrix
    homography, inliers = ransac(correspondences)
    print("Homography: \n", homography)
    display_correspondences(image1, image2, correspondences, inliers)

    # Warp one image onto the other and blend overlapping pixels
    output = warp_image(image1, image2, homography)

    # Save and display the output image
    cv2.imwrite("output_final.jpg", output)
    cv2.imshow("output final", output)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()