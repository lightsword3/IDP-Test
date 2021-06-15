import cv2
import numpy as np


def import_image_as_matrix(file_name: str) -> np.ndarray:
    """Import the specified png image and convert it to a boolean matrix"""

    # Read file
    im = cv2.imread(file_name)

    # To Grayscale
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    # To Black & White
    im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)[1]

    # Return 2d array
    return np.array(im, dtype=np.int16)


def output_matrix_as_image(file_name: str, matrix: np.ndarray):
    """Export the given matrix as an image."""
    cv2.imwrite(file_name, matrix)


if __name__ == "__main__":
    # Testing
    im = import_image_as_matrix('../data/test_420x400.png')
    output_matrix_as_image("../output/binary_input.png", im)
