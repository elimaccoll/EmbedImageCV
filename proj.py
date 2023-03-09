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


def NCC(img1, img2, window_size=5):
    p = int((window_size - 1) / 2)
    img1_cp = img1.copy()
    img1 = np.pad(img1, p, "constant", constant_values=0)
    ncc = np.zeros((img1.shape[0], img1.shape[1]))
    img1_hat = img1 / np.linalg.norm(img1)
    img2_hat = img2 / np.linalg.norm(img2)

    # print(img1_cp.shape)
    # print(img1_hat.shape)
    # print(img2_hat.shape)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if (i == 0 + p or i == img1_cp.shape[0] - 1) and (
                j == 0 + p or j == img1_cp.shape[1] - 1
            ):
                p1 = img1_hat[i - p : i + p + 1, j - p : j + p + 1]
                prod = np.mean((p1 - np.mean(p1)) * (img2_hat - np.mean(img2_hat)))
                std_div = np.std(img1_hat) * np.std(img2_hat)
                if std_div == 0:
                    ncc[i, j] = np.NaN
                else:
                    ncc[i, j] = prod / std_div
            else:
                ncc[i, j] = -np.infty
    return ncc


def NCC2(w, g, pad):
    w_pad = np.pad(w, pad, "constant", constant_values=0)

    g_hat = g / np.linalg.norm(g)
    w_hat = w_pad / np.linalg.norm(w_pad)

    ncc = np.zeros((w_pad.shape[0], w_pad.shape[1]))
    return convolve2d(w_hat, g_hat, mode="full")

    for i in range(w_pad.shape[0]):
        for j in range(w_pad.shape[1]):
            if (i == pad or i == w.shape[0] - 1) and (j == pad or j == w.shape[1] - 1):
                w1 = w_hat[i - pad : i + pad + 1, j - pad : j + pad + 1]
                prod = np.mean((w1 - np.mean(w1)) * (g_hat - np.mean(g_hat)))
                std_div = np.std(w_hat) * np.std(g_hat)
                if std_div == 0:
                    ncc[i, j] = np.NaN
                else:
                    ncc[i, j] = prod / std_div
            else:
                ncc[i, j] = -np.infty
    return ncc


def show(col, gray):
    for i in range(len(col)):
        cv2.imshow("{i}", col[i])
        cv2.imshow("G{i}", gray[i])
        if cv2.waitKey(0) & 0xFF == ord("q"):
            if 0xFF == ord("e"):
                break
            else:
                continue


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

    c1coords = list(zip(corner1[0], corner1[1]))
    c2coords = list(zip(corner2[0], corner2[1]))

    window_size = 5
    pad = int((window_size - 1) / 2)

    g_pad = np.pad(gray[1], pad, "constant", constant_values=0)

    for i, corner in enumerate(c2coords):
        f = g_pad[
            corner[0] + pad - 2 : corner[0] + pad + 3,
            corner[1] + pad - 2 : corner[1] + pad + 3,
        ]
        res = NCC2(gray[0], f, pad)
        maxVal = np.nanmax(res)
        maxLoc = np.where(res == maxVal)
        maxLoc = np.array(list(zip(maxLoc[0], maxLoc[1])))

        col[0][maxLoc[0][0], maxLoc[0][1]] = [0, 0, 255]
        col[1][corner[0], corner[1]] = [0, 0, 255]
        print(f"{i}/{len(c2coords)}", maxLoc[0][0], maxLoc[0][1], end="\r")

    show([col[0]], [col[1]])


if __name__ == "__main__":
    main()
