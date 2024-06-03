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

def interpolate_1_16(img, x, y, coeffs):
    """Performs 1/16-pel interpolation using the 8-tap filter coefficients."""
    ix = int(x)
    iy = int(y)
    fx = int((x - ix) * 16)  # Convert fractional part to index (0-15)
    fy = int((y - iy) * 16)

    value = 0.0
    for m in range(-3, 5):
        for n in range(-3, 5):
            px = np.clip(ix + m, 0, img.shape[1] - 1)
            py = np.clip(iy + n, 0, img.shape[0] - 1)
            value += img[py, px] * coeffs[fy][m + 3] * coeffs[fx][n + 3]

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
