import os
import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
import cv2
from eval_one_img import calculate_psnr_for_frame

def blur(img, kernel_size = 3):
	kernel = np.ones((3, 3), np.float32) / 9
	convolved_img = cv2.filter2D(img, -1, kernel)
	return convolved_img

def add_gaussian_noise(img, sigma):
	gauss = np.random.normal(0, sigma, np.shape(img))
	noisy_img = img + gauss
	noisy_img[noisy_img < 0] = 0
	noisy_img[noisy_img > 255] = 255
	return noisy_img

def wiener_filter(img, kernel, K):
	kernel /= np.sum(kernel)
	dummy = np.copy(img)
	dummy = fft2(dummy)
	kernel = fft2(kernel, s = img.shape)
	kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
	dummy = dummy * kernel
	dummy = np.abs(ifft2(dummy))
	return dummy

def gaussian_kernel(kernel_size = 3):
	sigma = 0.05
	kernel = cv2.getGaussianKernel(kernel_size, sigma)
	gau_kernel = np.outer(kernel, kernel)

	return gau_kernel

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def wiener_fliter():
	file_name = './results/DaylightRoad2.yuv/compensated/016.png'
	img = rgb2gray(cv2.imread(file_name))
	# Blur the image
	#blurred_img = blur(img, kernel_size = 15)

	# Add Gaussian noise
	#noisy_img = add_gaussian_noise(blurred_img, sigma = 20)

	# Apply Wiener Filter
	kernel = gaussian_kernel(3)
	filtered_img = wiener_filter(img, kernel, K = 0.5)

	return filtered_img
filtered=wiener_fliter()
img = cv2.imread('./results/DaylightRoad2.yuv/frames/016.png')
cv2.imwrite('./filtered.png',filtered)
print(filtered.shape)
##psnr=calculate_psnr_for_frame(filtered, img, './results/DaylightRoad2.yuv/sel_map/016.txt') 