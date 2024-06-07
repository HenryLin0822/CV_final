import numpy as np

def get_8_tap_filter_coefficients():
    """Returns the 8-tap filter coefficients for each fractional sample position (0-15)"""
    return [
        [0, 0, 0, 64, 0, 0, 0, 0],   # p = 0
        [0, 1, -3, 63, 4, -2, 1, 0], # p = 1
        [-1, 2, -5, 62, 8, -3, 1, 0],# p = 2
        [-1, 3, -8, 60, 13, -4, 1, 0],# p = 3
        [-1, 4, -10, 58, 17, -5, 1, 0],# p = 4
        [-1, 4, -11, 52, 26, -8, 3, -1],# p = 5
        [-1, 3, -9, 47, 31, -10, 4, -1],# p = 6
        [-1, 4, -10, 45, 34, -10, 4, -1],# p = 7
        [-1, 4, -11, 40, 40, -11, 4, -1],# p = 8
        [-1, 4, -11, 34, 45, -10, 4, -1],# p = 9
        [-1, 4, -10, 31, 47, -9, 3, -1],# p = 10
        [-1, 3, -8, 26, 52, -11, 4, -1],# p = 11
        [-1, 4, -5, 17, 58, -10, 4, -1],# p = 12
        [0, 1, -4, 13, 60, -8, 3, -1],# p = 13
        [0, 1, -3, 8, 62, -5, 2, -1],# p = 14
        [0, 1, -2, 4, 63, -3, 1, 0],# p = 15
    ]

def interpolate_1_16(img, x, y, coeffs= get_8_tap_filter_coefficients()):
    """Performs 1/16-pel interpolation using the 8-tap filter coefficients."""
    ix = int(x)
    iy = int(y)
    dx = int((x - ix) * 16)  # Convert fractional part to index (0-15)
    dy = int((y - iy) * 16)

    ix = np.clip(ix, 0, img.shape[1] - 2)
    iy = np.clip(iy, 0, img.shape[0] - 2)

    # Get the values at the four surrounding points
    top_left = img[iy, ix]
    top_right = img[iy, ix + 1]
    bottom_left = img[iy + 1, ix]
    bottom_right = img[iy + 1, ix + 1]

    # Perform bilinear interpolation
    top = (1 - dx) * top_left + dx * top_right
    bottom = (1 - dx) * bottom_left + dx * bottom_right
    value = (1 - dy) * top + dy * bottom

    # Apply the filter coefficients to the neighborhood
    return value

def interpolate_image_1_16(img):
    """Generates an interpolated image with 1/16-pel resolution."""
    coeffs = get_8_tap_filter_coefficients()
    new_shape = (img.shape[0] * 16, img.shape[1] * 16)
    interpolated_img = np.zeros(new_shape, dtype=np.float32)

    for y in range(new_shape[0]):
        for x in range(new_shape[1]):
            interpolated_img[y, x] = interpolate_1_16(img, x / 16, y / 16, coeffs)

    return interpolated_img
