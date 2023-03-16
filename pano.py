import os
from random import choice, choices, randint

from typing import Tuple

import cv2
import numpy as np


class Panorama:
    def __init__(self, path):
        if path == "DanaHallWay1":
            self.harris_thresh = 0.0025
            self.ncc_thresh = 0.9995
        elif path == "DanaOffice":
            self.harris_thresh = 0.01
            self.ncc_thresh = 0.9995 # 0.95
        return

    def load_images(self, path: str) -> np.ndarray:
        """Load images from a path.

        Args:
            path (string): Path to the folder containing the images.

        Returns:
            np.ndarray: Color images.
            np.ndarray: Grayscale images.
        """
        files = [
            os.path.join(path, f)
            for f in sorted(os.listdir(path))
            if f.endswith(".JPG")
        ]
        return np.array([cv2.imread(f) for f in files]), np.array(
            [cv2.imread(f, 0) for f in files]
        )

    def harris_corner_detector(
        self,
        image: np.ndarray,
        k: float = 0.04,
        window_size: int = 3
    ) -> np.ndarray:
        """Given an image, it finds the corners using the Harris Corner Detector.

        Args:
            image (np.ndarray): Grayscale image to find the corners.
            k (float, optional): Empirically determined constant in range 0.04 <= x <= 0.06. Defaults to 0.04.
            window_size (int, optional): The search window size for operations such as Gaussian Blur, Sobel Mask etc.. Defaults to 3.
        Returns:
            corners (list): Corners in the image in the form [(x, y)].
        """
        # Gaussian Blur over the image
        image = cv2.GaussianBlur(image, (window_size, window_size), 0)

        # Compute the gradients
        Ix = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=window_size)
        Iy = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, ksize=window_size)

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

        # Thresholding
        R[R > self.harris_thresh] = 255

        # Non Max Suppression
        R[R <= self.harris_thresh] = 0

        corners = np.where(R == 255)
        corners = list(zip(corners[0], corners[1]))
        return corners

    def normalized_cross_correlation(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        corners1: list,
        corners2: list,
        window_size: int = 5
    ) -> dict:
        """Given two images and their corners, it computes the normalized cross correlation between the two images and returns the correspondences.

        Args:
            image1 (np.ndarray): Color image.
            image2 (np.ndarray): Color image.
            corners1 (list): It is a list of tuples (x, y) where x and y are the coordinates of the corners in image1.
            corners2 (list): It is a list of tuples (x, y) where x and y are the coordinates of the corners in image2.
            window_size (int, optional): Window Size. Defaults to 5.

        Returns:
            Correspondences (dict): Correspondences between the two images in the form {corner1 (x, y): corner2 (x, y)}
        """
        pad = window_size // 2

        # Pad gray images
        image1_pad = np.pad(image1, pad, "constant", constant_values=0)
        image2_pad = np.pad(image2, pad, "constant", constant_values=0)

        correspondences = {}
        for i, corner1 in enumerate(corners1):
            # Image patch around corner 1
            x1 = max(corner1[0] - pad, 0)
            x2 = max(corner1[0] + pad + 1, window_size)
            y1 = max(corner1[1] - pad, 0)
            y2 = max(corner1[1] + pad + 1, window_size)
            patch1 = image1_pad[
                x1 : x2,
                y1 : y2,
            ]

            max_ncc = -1
            best_corner = None
            for j, corner2 in enumerate(corners2):
                print(f"i={i + 1}/{len(corners1)} j={j + 1}/{len(corners2)}", end="\r")

                # Image patch around corner 2
                x1 = max(corner2[0] - pad, 0)
                x2 = max(corner2[0] + pad + 1, window_size)
                y1 = max(corner2[1] - pad, 0)
                y2 = max(corner2[1] + pad + 1, window_size)
                patch2 = image2_pad[
                    x1 : x2,
                    y1 : y2,
                ]
                # Calculate NCC using image patches
                patch1_hat = patch1 / np.linalg.norm(patch1)
                patch2_hat = patch2 / np.linalg.norm(patch2)
                ncc = np.sum(patch1_hat * patch2_hat)

                # If this NCC is the new max, store it and the coords of the corner
                if ncc > max_ncc:
                    max_ncc = ncc
                    best_corner = corner2

                # Threshold
                if max_ncc < self.ncc_thresh:
                    continue

                # Store correspondence with highest NCC
                correspondences[corner1] = best_corner
        return correspondences

    def ransac(
        self, correspondences: dict, threshold: int = 5, k: int = 100, N: int = 4
    ) -> np.ndarray:
        """Performs RANSAC to find the best homography matrix, given the correspondences. This is done using the largest set of inliers.

        Args:
            correspondences (dict): Correspondences between the two images in the form {corner1 (x, y): corner2 (x, y)}
            threshold (int, optional): The distance threshold. Defaults to 5.
            k (int, optional): No of iterations. Defaults to 100.
            N (int, optional): Sample size. Defaults to 4.

        Returns:
            H (np.ndarray): Homography matrix.
            inliers (dict): Inliers in the form {corner1 (x, y): corner2 (x, y)}.
            outliers (dict): Outliers in the form {corner1 (x, y): corner2 (x, y)}.
        """
        H = None
        inliers = {}
        outliers = {}
        max_inliers = 0

        for _ in range(k):
            set_inliers = {}
            set_outliers = {}

            num_inliers = 0
            num_outliers = 0

            # Sample 4 points from the correspondences
            points1 = choices(list(correspondences.keys()), k=4)
            points2 = [tuple(correspondences.get(p)) for p in points1]

            # Compute homography matrix using these points
            h, status = cv2.findHomography(np.asarray(points1), np.asarray(points2))
            
            for corner1, corner2 in correspondences.items():
                # Estimate point using homography
                pt1 = np.array([corner1[0], corner1[1], 1])
                pt2 = np.array([corner2[0], corner2[1], 1])
                res = np.dot(h, pt1)
                # res = np.matmul(h, pt1)
                res = (res[:2]/res[2]).astype(int)
                dist = np.linalg.norm(res - corner2)

                # Check if outlier
                if dist > threshold:
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
                H = h

        return H, inliers, outliers

    def create_panorama(
        self, image1: np.ndarray, image2: np.ndarray, H: np.ndarray
    ) -> np.ndarray:
        """Creates a panorama image given two images and the homography matrix.

        Args:
            image1 (np.ndarray): Color image.
            image2 (np.ndarray): Color image.
            H (np.ndarray): Homography matrix, H (3x3).

        Returns:
            Final (np.ndarray): Final panorama image.
        """
        copy1, copy2 = image1.copy(), image2.copy()
        stitcher = cv2.Stitcher().create()
        final_size = (copy1.shape[1] + copy2.shape[1], copy2.shape[0])
        final = cv2.warpPerspective(copy2, H, final_size)
        _, final = stitcher.stitch((copy1, copy2))
        return final

    def drawLines(self, image1: np.ndarray, image2: np.ndarray, points: dict, col1: tuple = None) -> None:
        """Draws lines between the points in the two images.

        Args:
            image1 (np.ndarray): Color image.
            image2 (np.ndarray): Color image.
            points (dict): dict of points in the form {corner1 (x, y): corner2 (x, y)}.
        Return:
            vis (np.ndarray): Color image with lines.
        """
        # Concatenate the two images
        vis = np.concatenate((image1, image2), axis=1)

        # Draw lines between correlated corners
        for pt1, pt2 in points.items():
            col = col1 if col1 else (randint(0, 255), randint(0, 255), randint(0, 255))
            start_y, start_x = pt1
            end_y, end_x = pt2
            end_x = int(end_x + vis.shape[1]/2)
            end = (end_x, end_y)
            start = (start_x, start_y)
            cv2.line(vis, start, end, col, 1)

        return vis
    
    def drawCorners(self, image: np.ndarray, corners: list) -> None:
        """Draw corners in the image.

        Args:
            image (np.ndarray): Color image.
            corners (list): List of points in the form [(x, y)].
        Returns:
            vis (np.ndarray): Color image with corners drawn.
        """
        vis = image.copy()
        # Draw lines between correlated corners
        for corner in corners:
            cv2.circle(vis, (corner[1], corner[0]), 1, (255, 0, 0), 1)
        return vis

    def show_image(self, images: list, titles: list) -> None:
        """Shows a list of images along with their titles.

        Args:
            images (list): A list of all the images to be shown.
            titles (list): A list of all the titles of the images.
        """
        for image, title in zip(images, titles):
            cv2.imshow(title, image)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyAllWindows()

    def save_image(self, images: list, fnames: list) -> None:
        """Saves a list of images with their corresponding filename

        Args:
            images (list): A list of all images to be saved
            fnames (list): A list of all the file names of the images
        """
        for image, fname in zip(images, fnames):
            cv2.imwrite(f"results/{fname}.jpg", image)



def main():
    # DIR = "DanaHallWay1"
    DIR = "DanaOffice"

    pano = Panorama(DIR)
    
    # Load images
    col, gray = pano.load_images(DIR)
    image1, image2 = col[0], col[1]
    gray1, gray2 = gray[0], gray[1]

    # Apply Harris corner detector
    corners1 = pano.harris_corner_detector(gray1)
    corners2 = pano.harris_corner_detector(gray2)

    # samples = 600
    # corners1 = choices(corners1, k=samples)
    # corners2 = choices(corners2, k=samples)

    # Find correspondences using NCC
    correspondences = pano.normalized_cross_correlation(gray1, gray2, corners1, corners2)

    # Use RANSAC to estimate homography matrix and find inliers
    H, inliers, outliers = pano.ransac(correspondences)

    # Warp images
    output = pano.create_panorama(image1, image2, H)

    # Display results
    corners1_vis = pano.drawCorners(image1, corners1)
    corners2_vis = pano.drawCorners(image2, corners2)
    correspondences_vis = pano.drawLines(image1, image2, correspondences)
    inliers_vis = pano.drawLines(image1, image2, inliers)
    outliers_vis = pano.drawLines(image1, image2, outliers)
    
    pano.show_image([image1, image2, corners1_vis, corners2_vis, correspondences_vis, inliers_vis, outliers_vis, output], ["Input 1", "Input 2", "corners1", "corners2", "correspondences", "inliers", "outliers", "Output"])

    # Save results
    pano.save_image([image1, image2, corners1_vis, corners2_vis, correspondences_vis, inliers_vis, outliers_vis, output], [f"{DIR}_input1", f"{DIR}_input2", f"{DIR}_corners1", f"{DIR}_corners2", f"{DIR}_correspondences", f"{DIR}_inliers", f"{DIR}_outliers", f"{DIR}_output"])

if __name__ == "__main__":
    main()