import os
import cv2
import numpy as np

# Modified code from pano.py to just contain the image embedding functionality

class Embed:
    def __init__(self) -> None:
        """Initialize the Panorama class."""
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
            # if f.endswith(".JPG")
        ]
        return np.array([cv2.imread(f) for f in files]), np.array(
            [cv2.imread(f, 0) for f in files]
        )

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
    """Embeds an image into another image."""

    embed = Embed()

    # Load images
    base_images, _ = embed.load_images("base")
    embed_images, _ = embed.load_images("embed")

    embed_image = embed_images[0]
    base_image = base_images[0]

    # Warp an image into a region in the second image
    output = embed.embed_image(embed_image, base_image)

    # Display results
    embed.show_image([output], ["embed_output"])

    # Save Results
    embed.save_image([output], ["embed_output"])


if __name__ == "__main__":
    main()
