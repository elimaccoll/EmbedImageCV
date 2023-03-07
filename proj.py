# Image Mosiac Project

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import os
import cv2


# Load the images
def load_images(DIR):
    fn = [os.path.join(DIR, f) for f in sorted(os.listdir(DIR)) if f.endswith(".JPG")]
    return [cv2.imread(f) for f in fn], [cv2.imread(f, 0) for f in fn]


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
    Rd = cv2.dilate(R, None)

    # Thresholding
    R[Rd > threshold] = 255

    # Non Max Suppression
    R[Rd <= threshold] = 0

    return R


def show(col, gray):
    for i in range(len(col)):
        cv2.imshow(f"{i}", col[i])
        cv2.imshow(f"G{i}", gray[i])
        if cv2.waitKey(0) & 0xFF == ord("q"):
            if 0xFF == ord("e"):
                break
            else:
                continue


if __name__ == "__main__":
    DIR = "DanaHallWay1"
    # DIR = "DanaOffice"
    col, gray = load_images(DIR)
    harris_corner_detector(gray[0], col[0])
