import os
from random import choices, randint

import cv2
import numpy as np


class Panorama:
    def __init__(self, path: str) -> None:
        """Initialize the Panorama class.

        Args:
            path (str): Path to the folder containing the images.
        """
        if path == "DanaHallWay1":
            self.harris_thresh = 0.0025
        elif path == "DanaOffice":
            self.harris_thresh = 0.01

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
            # if f.endswith(".JPG")
        ]
        return np.array([cv2.imread(f) for f in files]), np.array(
            [cv2.imread(f, 0) for f in files]
        )

    def non_maximum_suppresion(
        self, image: np.ndarray, window_size: int = 5
    ) -> np.ndarray:
        """Apply non-maximum suppression to a given image

        Args:
            image (np.ndarray): Image to perform non-max suppression on the.
            window_size (int, optional): The window size around each pixel.
        Returns:
            suppressed (np.ndarray): Resulting image after non-maximum suppression.
        """
        suppressed = image.copy()
        global_min = image.min()
        p = window_size // 2

        width, height = suppressed.shape

        for i in range(width):
            x1 = max(0, i - p)
            x2 = min(width, i + p)
            for j in range(height):
                # Bounds for window around pixel at (i, j)
                y1 = max(0, j - p)
                y2 = min(height, j + p)

                # Set pixel value to the global min to exclude it from max
                value = suppressed[i, j]
                suppressed[i, j] = global_min

                # If pixel has the maximum value in the window then use its value
                local_max = suppressed[x1:x2, y1:y2].max()
                if value > local_max:
                    suppressed[i, j] = value

                # Else keep it is global minimum
        return suppressed

    def harris_corner_detector(
        self, image: np.ndarray, k: float = 0.04, window_size: int = 3
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

        # Non-max suppression
        R = self.non_maximum_suppresion(R)

        # Thresholding
        R[R > self.harris_thresh] = 255

        corners = np.where(R == 255)
        corners = list(zip(corners[0], corners[1]))
        return corners

    def normalized_cross_correlation(
        self,
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
                x1:x2,
                y1:y2,
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
                    x1:x2,
                    y1:y2,
                ]
                # Calculate NCC using image patches
                patch1_hat = patch1 / np.linalg.norm(patch1)
                patch2_hat = patch2 / np.linalg.norm(patch2)
                ncc = np.sum(patch1_hat * patch2_hat)

                # If this NCC is the new max, store it and the coords of the corner
                if ncc > max_ncc:
                    max_ncc = ncc
                    best_corner = corner2

                # Store correspondence with highest NCC
                correspondences[corner1] = best_corner
        return correspondences

    def homography(self, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """Given two sets of points, it computes the homography matrix.

        Args:
            points1 (np.ndarray): Points in the form [[x1, y1], [x2, y2], ...] of shape (4 x 2).
            points2 (np.ndarray): Points in the form [[x1, y1], [x2, y2], ...] of shape (4 x 2).

        Returns:
            h_mat (np.ndarray): Homography matrix of shape (3 x 3).
        """
        H = np.zeros((points1.shape[0] * 2, 9))
        for i, ((x1, y1), (x2, y2)) in enumerate(zip(points1, points2)):
            # print(f"{i}, (({x1}, {y1}), ({x2}, {y2}))")
            H[2 * i] = [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]
            H[2 * i + 1] = [0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2]
        _, _, V = np.linalg.svd(H.T @ H)
        h = V[-1]
        h_mat = h.reshape(3, 3)
        h_mat = h_mat / h_mat[2, 2]
        return h_mat

    def ransac(
        self, correspondences: dict, threshold: int = 5, k: int = 72, N: int = 4
    ) -> np.ndarray:
        """Performs RANSAC to find the best homography matrix, given the correspondences. This is done using the largest set of inliers.

        Args:
            correspondences (dict): Correspondences between the two images in the form {corner1 (x, y): corner2 (x, y)}
            threshold (int, optional): The distance threshold. Defaults to 5.
            k (int, optional): No of iterations. Defaults to 72.
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
            h = self.homography(np.asarray(points1), np.asarray(points2))

            for corner1, corner2 in correspondences.items():
                # Estimate point using homography
                pt1 = np.array([corner1[0], corner1[1], 1])
                pt2 = np.array([corner2[0], corner2[1], 1])
                res = np.dot(h, pt1)
                # res = np.matmul(h, pt1)
                res = (res[:2] / res[2]).astype(int)
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

    def embed_image(
        self, embed_image: np.ndarray, base_image: np.ndarray
    ) -> np.ndarray:
        """Embeds an image into a given base image.

        Args:
            embed_image (np.ndarray): The image being embedded in the base image.
            base_image (np.ndarray): The base image being embedded into.
        Returns:
            composite_image (np.ndarray): The resulting composite image containing the embed.
        """
        # Create a copy of the base image
        base_image = base_image.copy()

        # Get the desired region to embed image on reom mouse events.
        region = self.get_user_region(base_image)
        region = np.asarray(region)

        base_width, base_height, _ = base_image.shape
        embed_width, embed_height, _ = embed_image.shape

        # Clockwise starting from top left corner
        embed_image_corners = np.array(
            [(0, 0), (embed_width, 0), (embed_width, embed_height), (0, embed_height)]
        )

        # Calculate homography matrix
        H = self.homography(embed_image_corners, region)

        # Warp the image into the user selected region using the homography matrix
        embed_image_isolated = cv2.warpPerspective(
            embed_image, H, (base_height, base_width)
        )

        # Create a mask of the user selected region
        fill_mask = np.zeros(base_image.shape).astype("uint8")
        cv2.fillConvexPoly(fill_mask, region, (255, 255, 255))

        # Apply mask of user selected region to the base image
        base_image_with_region = cv2.bitwise_and(base_image, cv2.bitwise_not(fill_mask))

        # Add the base image and the isolated embed image
        composite_image = base_image_with_region + embed_image_isolated

        return composite_image

    def mouse_callback(
        self, event: int, x: int, y: int, flags: int, param: dict
    ) -> None:
        """Callback function for mouse events used when obtaining the user selected region for embedding an image.

        Args:
            event (int): The type of mouse click event.
            x (int): The x pixel coordinate of the mouse in base image.
            y (int): The y pixel coordinate of the mouse in base image.
            flags (int): Event flags.
            param (dict): Any parameters that are passed in.
        """

        # Handle when the mouse is clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get image and corners by reference
            base_image = param["image"]
            corners = param["corners"]
            corners.append([x, y])
            # Draw square centered around click
            p = 5
            start_point = (x - p, y - p)  # Top left corner
            end_point = (x + p, y + p)  # Bottom right corner
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(base_image, start_point, end_point, color, thickness)

    def get_user_region(self, base_image: np.ndarray) -> list:
        """Get the corners of the region to embed the image into from the user using mouse click on the base image.

        Args:
            base_image (np.ndarray): The base image being embedded onto.
        Returns:
            corners (list): the pixel coordinates of the corners of the user selected region in the form [(x, y)]
        """
        corners = []
        param = {
            "image": base_image,
            "corners": corners,
        }

        window_name = "Embed Image"
        cv2.imshow(window_name, base_image)
        cv2.setMouseCallback(window_name, self.mouse_callback, param)

        while True:
            cv2.imshow(window_name, base_image)
            if cv2.waitKey(1) & 0xFF == ord("q") or len(corners) == 4:
                cv2.destroyWindow(window_name)
                break

        return corners

    def draw_lines(
        self, image1: np.ndarray, image2: np.ndarray, points: dict, col1: tuple = None
    ) -> None:
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
            end_x = int(end_x + vis.shape[1] / 2)
            end = (end_x, end_y)
            start = (start_x, start_y)
            cv2.line(vis, start, end, col, 1)

        return vis

    def draw_corners(self, image: np.ndarray, corners: list) -> None:
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
            cv2.circle(vis, (corner[1], corner[0]), 3, (0, 0, 255), 1)
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
    """Main function to run the Panorama class."""
    # The directory to use two images from to create a mosaic
    DIR = "DanaHallWay1"
    # DIR = "DanaOffice"
    pano = Panorama(DIR)

    # Load images
    col, gray = pano.load_images(DIR)
    image1, image2 = col[0], col[1]
    gray1, gray2 = gray[0], gray[1]

    # Apply Harris corner detector
    corners1 = pano.harris_corner_detector(gray1)
    corners2 = pano.harris_corner_detector(gray2)

    # Find correspondences using NCC
    correspondences = pano.normalized_cross_correlation(
        gray1, gray2, corners1, corners2
    )

    # Use RANSAC to estimate homography matrix and find inliers
    H, inliers, outliers = pano.ransac(correspondences)

    # Warp images
    output = pano.create_panorama(image1, image2, H)

    # Create visualiations of results
    corners1_vis = pano.draw_corners(image1, corners1)
    corners2_vis = pano.draw_corners(image2, corners2)
    correspondences_vis = pano.draw_lines(image1, image2, correspondences)
    inliers_vis = pano.draw_lines(image1, image2, inliers)
    outliers_vis = pano.draw_lines(image1, image2, outliers)

    # Display results
    print(H)
    pano.show_image(
        [
            image1,
            image2,
            corners1_vis,
            corners2_vis,
            correspondences_vis,
            inliers_vis,
            outliers_vis,
            output,
        ],
        [
            "Input 1",
            "Input 2",
            "corners1",
            "corners2",
            "correspondences",
            "inliers",
            "outliers",
            "Output",
        ],
    )

    # Save results
    pano.save_image(
        [
            image1,
            image2,
            corners1_vis,
            corners2_vis,
            correspondences_vis,
            inliers_vis,
            outliers_vis,
            output,
        ],
        [
            f"{DIR}_input1",
            f"{DIR}_input2",
            f"{DIR}_corners1",
            f"{DIR}_corners2",
            f"{DIR}_correspondences",
            f"{DIR}_inliers",
            f"{DIR}_outliers",
            f"{DIR}_output",
        ],
    )


def extra_credit():
    """Extra credit function that embeds an image into another image."""
    # The directory to pull an image from to use as the base image
    DIR = "DanaHallWay1"
    # DIR = "DanaOffice"

    pano = Panorama(DIR)

    # Load images
    col, _ = pano.load_images(DIR)
    embed_images, _ = pano.load_images("ec")

    embed_image = embed_images[0]
    # embed_image2 = embed_images[1]
    base_image = col[0]

    # Warp an image into a region in the second image
    output = pano.embed_image(embed_image, base_image)

    # Display results
    pano.show_image([output], ["ExtraCredit_output"])

    # Save Results
    pano.save_image([output], ["ExtraCredit_output"])


if __name__ == "__main__":
    main()
    extra_credit()
