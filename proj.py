from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import os
import cv2
from random import randint, choice, choices
from math import sqrt
from skimage import transform

# Load the images
def load_images(DIR):
    fn = [os.path.join(DIR, f) for f in sorted(os.listdir(DIR)) if f.endswith(".JPG")]
    return np.array([cv2.imread(f) for f in fn]), np.array(
        [cv2.imread(f, 0) for f in fn]
    )

# Harris Corner Detection
def harris_corner_detector(img, colimg, k=0.04, window_size=3, threshold=0.0025):
    # Gaussian Blur over the image
    img = cv2.GaussianBlur(img, (window_size, window_size), 0)

    # Compute the gradients
    Ix = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=window_size)
    Iy = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=window_size)

    # Compute the products of the gradients
    IxIy = Ix * Iy
    Ix2 = Ix**2
    Iy2 = Iy**2

    # Compute the sums of the products of the gradients
    S2x = cv2.GaussianBlur(Ix2, (window_size, window_size), 0)
    S2y = cv2.GaussianBlur(Iy2, (window_size, window_size), 0)
    Sxy = cv2.GaussianBlur(IxIy, (window_size, window_size), 0)

    # Harris Corner Response
    det = (S2x * S2y) - (Sxy**2)
    trace = S2x + S2y

    R = det - k * (trace**2)

    # Normalize
    R /= R.max()

    # Dilate
    # Rd = cv2.dilate(R, None)
    Rd = R

    # Thresholding
    R[Rd > threshold] = 255

    # Non Max Suppression
    R[Rd <= threshold] = 0

    return R

def NCC(f, g):
    f_hat = f / np.linalg.norm(f)
    g_hat = g / np.linalg.norm(g)

    ncc = np.sum(f_hat * g_hat)
    return ncc

def show(col, gray):
    for i in range(len(col)):
        cv2.imshow("{i}", col[i])
        cv2.imshow("G{i}", gray[i])
        if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            if 0xFF == ord("e"):
                break
            else:
                continue

# corners: dict[tuple[int, int]: tuple[int, int]]
def drawCorrelationLines(image1, image2, corners):
    # Concatenate the two images
    vis = np.concatenate((image1, image2), axis=1)

    # Draw lines between correlated corners
    for key, value in corners.items():
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        start_y, start_x = key
        end_y, end_x = value
        end_x = int(end_x + vis.shape[1]/2)
        end = (end_x, end_y)
        start = (start_x, start_y)
        cv2.line(vis, start, end, (b, g, r), 1)

    show([vis], [vis])

def dist(a, b):
    return np.linalg.norm(a-b)

def RANSAC(corners):
    k = 100
    N = 4

    inliers = {}
    outliers = {}
    max_inliers = 0

    # TODO: This is kinda arbitrary - trial and error maybe
    dist_threshold = 4

    for i in range(k):
        # print(f"===== Iteration #{i + 1}")
        points1 = []
        points2 = []

        set_inliers = {}
        set_outliers = {}

        num_inliers = 0
        num_outliers = 0

        # Sample the minimal number of points needed to estimate a homography (4 pts)
        for _ in range(N):
            pt1 = choice(list(corners.keys()))
            pt2 = list(corners[pt1])
            pt1 = list(pt1)
            points1.append(pt1)
            points2.append(pt2)
        points1 = np.asarray(points1)
        points2 = np.asarray(points2)

        # Compute homography matrix using these points
        h, status = cv2.findHomography(points1, points2)
        try:
            h_inv = np.linalg.pinv(h)
        except:
            continue

        for corner1, corner2 in corners.items():
            # print(corner1, corner2)
            # Estimate corner1 using homography
            mat1 = np.array([corner2[0], corner2[1], 1])
            res1 = np.matmul(h_inv, mat1)
            res1 = (res1[:2]/res1[2]).astype(int)
            dist1 = dist(res1, corner1)

            # Estimate corner2 using homography
            mat2 = np.array([corner1[0], corner1[1], 1])
            res2 = np.matmul(h, mat2)
            res2 = (res2[:2]/res2[2]).astype(int)
            dist2 = dist(res2, corner2)

            # Check if outlier
            if dist1 > dist_threshold or dist2 > dist_threshold:
                set_outliers[tuple(corner1)] = tuple(corner2)
                num_outliers += 1
                continue

            # Store inlier
            set_inliers[tuple(corner1)] = tuple(corner2)
            num_inliers += 1

        # Check if this homography produced the new largest set of inliers
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            inliers = set_inliers
            outliers = set_outliers

    # print(max_inliers, inliers)
    return inliers


def main():
    DIR = "DanaHallWay1"  # --> k=0.0025
    # DIR = "DanaOffice" #--> k=0.01 (?)
    col, gray = load_images(DIR)

    harris = []
    for i in range(len(col)):
        R1 = harris_corner_detector(gray[i], col[i])
        R2 = harris_corner_detector(gray[i + 1], col[i + 1])
        break

    corner1 = np.where(R1 == 255)
    corner2 = np.where(R2 == 255)

    # List of coordinates of corners
    c1coords = list(zip(corner1[0], corner1[1]))
    c2coords = list(zip(corner2[0], corner2[1]))

    # Window size and padding for NCC
    window_size = 5
    pad = int((window_size - 1) / 2)

    # Pad gray images
    g1_pad = np.pad(gray[0], pad, "constant", constant_values=0)
    g2_pad = np.pad(gray[1], pad, "constant", constant_values=0)

    # Perform NCC
    # - Compare every corner detected in image 1 with every corner detected in image 2
    # - Store the pairs of corners with the highest correlation value
    corners = {}
    for i, corner1 in enumerate(c1coords):
        # Image patch around corner 1
        x1 = max(corner1[0] - pad, 0)
        x2 = max(corner1[0] + pad + 1, window_size)
        y1 = max(corner1[1] - pad, 0)
        y2 = max(corner1[1] + pad + 1, window_size)
        patch1 = g1_pad[
            x1 : x2,
            y1 : y2,
        ]

        maxNCC = -1
        coords = None
        for j, corner2 in enumerate(c2coords):
            print(f"i={i}/{len(c1coords)} j={j}/{len(c2coords)}", end="\r")

            # Image patch around corner 2
            x1 = max(corner2[0] - pad, 0)
            x2 = max(corner2[0] + pad + 1, window_size)
            y1 = max(corner2[1] - pad, 0)
            y2 = max(corner2[1] + pad + 1, window_size)
            patch2 = g2_pad[
                x1 : x2,
                y1 : y2,
            ]
            # Calculate NCC using image patches
            ncc = NCC(patch1, patch2)
            # If this NCC is the new max, store it and the coords of the corner
            if ncc > maxNCC:
                maxNCC = ncc
                coords = corner2[0], corner2[1]

        # Break earlier for testing
        # if i > 100:
        #     break

        # Threshold
        if maxNCC < 0.9995:
            continue

        # Store corner pair with highest NCC
        corners[(corner1[0], corner1[1])] = coords
        # Color the corners in the images
        # col[0][corner1[0], corner1[1]] = [0, 0, 255]
        # col[1][coords[0], coords[1]] = [0, 0, 255]


    inliers = RANSAC(corners)

    # Compute least-squares homography from ALL the inliers in the largest set of inliers
    points1 = np.array([*inliers.keys()])
    points2 = np.array([*inliers.values()])
    H, status = cv2.findHomography(points1, points2)

    drawCorrelationLines(col[0], col[1], inliers)

if __name__ == "__main__":
    main()