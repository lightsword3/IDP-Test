import image_interface as ii
import numpy as np


def shift_2d(matrix: np.ndarray, shift_x: int, shift_y: int):
    """Shift the binary matrix in x/y direction

    Boundary condition - fill with white
    Other sensible options: fill with black | repeat boundary
    """

    # init empty matrix of same dimensions
    result = np.empty_like(matrix)

    if shift_x > 0:
        result[:shift_x, :] = 255  # fill with white on shift
        if shift_y > 0:
            # x+, y+
            result[:, :shift_y] = 255  # fill with white on shift
            result[shift_x:, shift_y:] = matrix[:-shift_x, :-shift_y]
        elif shift_y < 0:
            # x+, y-
            result[:, shift_y:] = 255  # fill with white on shift
            result[shift_x:, :shift_y] = matrix[:-shift_x, -shift_y:]
        else:
            # x+, 0
            result[shift_x:, :] = matrix[:-shift_x, :]
    elif shift_x < 0:
        result[shift_x:, :] = 255  # fill with white on shift
        if shift_y > 0:
            # x-, y+
            result[:, :shift_y] = 255  # fill with white on shift
            result[:shift_x, shift_y:] = matrix[-shift_x:, :-shift_y]
        elif shift_y < 0:
            # x-, y-
            result[:, shift_y:] = 255  # fill with white on shift
            result[:shift_x, :shift_y] = matrix[-shift_x:, -shift_y:]
        else:
            # x-, 0
            result[:shift_x, :] = matrix[-shift_x:, :]
    else:
        if shift_y > 0:
            # 0, y+
            result[:, :shift_y] = 255  # fill with white on shift
            result[:, shift_y:] = matrix[:, :-shift_y]
        elif shift_y < 0:
            # 0, y-
            result[:, shift_y:] = 255  # fill with white on shift
            result[:, :shift_y] = matrix[:, -shift_y:]
        else:
            # 0, 0
            result[:, :] = matrix

    return result


def detect_borders(image: np.ndarray, is_8neighbor_method: bool) -> np.ndarray:

    # get black pixels with white 4-neighbors
    white_up = np.clip(image - shift_2d(image, 1, 0) + 255, 0, 255)
    white_down = np.clip(image - shift_2d(image, -1, 0) + 255, 0, 255)
    white_left = np.clip(image - shift_2d(image, 0, 1) + 255, 0, 255)
    white_right = np.clip(image - shift_2d(image, 0, -1) + 255, 0, 255)

    # union of all black pixels with white 4-neighbors
    borders_4neighbor = np.clip(
        white_up + white_down + white_left + white_right - 3 * 255, 0, 255)

    if is_8neighbor_method:
        # get black pixels with white diagonal neighbors
        white_ul = np.clip(image - shift_2d(image, 1, 1) + 255, 0, 255)
        white_ur = np.clip(image - shift_2d(image, 1, -1) + 255, 0, 255)
        white_dl = np.clip(image - shift_2d(image, -1, 1) + 255, 0, 255)
        white_dr = np.clip(image - shift_2d(image, -1, -1) + 255, 0, 255)

        # union of all black pixels with white 8-neighbors
        return np.clip(
            borders_4neighbor + white_ul + white_ur +
            + white_dl + white_dr - 4 * 255, 0, 255)
    else:
        return borders_4neighbor


def test():
    # img = ii.import_image_as_matrix('../data/test_8x8.png')
    img = ii.import_image_as_matrix('../data/test_420x400.png')

    # print(shift_2d(img, 0, 0))  # original

    # print(shift_2d(img, 1, 0))  # shift down
    # print(shift_2d(img, 0, -1))  # shift left
    # print(shift_2d(img, -1, 1))  # shift up,right

    # black with white above
    # print(np.clip(img - shift_2d(img, 1, 0) + 255, 0, 255))

    # print(detect_borders(img, is_8neighbor_method=False))
    # print(detect_borders(img, is_8neighbor_method=True))

    ii.output_matrix_as_image(
        '../output/borders_4n.png', detect_borders(img, False))
    ii.output_matrix_as_image(
        '../output/borders_8n.png', detect_borders(img, True))


if __name__ == "__main__":
    test()
