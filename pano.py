import os
from random import choice, choices, randint

import cv2
import numpy as np


class Panorama:
    def __init__(self):
        return

    def load_images(self, path: str) -> tuple(np.ndarray, np.ndarray):
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
        image: np.ndarray,
        k: float = 0.04,
        window_size: int = 3,
        threshold: float = 0.0025,
    ) -> np.ndarray:
        """Given an image, it finds the corners using the Harris Corner Detector.

        Args:
            image (np.ndarray): Grayscale image to find the corners.
            k (float, optional): Empirically determined constant in range 0.04 <= x <= 0.06. Defaults to 0.04.
            window_size (int, optional): The search window size for operations such as Gaussian Blur, Sobel Mask etc.. Defaults to 3.
            threshold (float, optional): Threshold value for determining if the point detected is an actual point. Defaults to 0.0025.

        Returns:
            R (np.ndarray): returns the response of the detector.
        """
        raise NotImplementedError

    def normalized_cross_correlation(
        image1: np.ndarray,
        image2: np.ndarray,
        corners1: list,
        corners2: list,
        window_size: int = 5,
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
        raise NotImplementedError

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
        raise NotImplementedError

    def drawLines(self, image1: np.ndarray, image2: np.ndarray, points: dict) -> None:
        """Draws lines between the points in the two images.

        Args:
            image1 (np.ndarray): Color image.
            image2 (np.ndarray): Color image.
            points (dict): dict of points in the form {corner1 (x, y): corner2 (x, y)}.
        """
        raise NotImplementedError

    def show_image(self, images: list, titles: list) -> None:
        """Shows a list of images along with their titles.

        Args:
            images (list): A list of all the images to be shown.
            titles (list): A list of all the titles of the images.
        """
        raise NotImplementedError

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
