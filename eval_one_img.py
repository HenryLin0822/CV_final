import os
import argparse
import numpy as np
from PIL import Image
import cv2

def calculate_psnr_for_frame(compensated, current, so_txt_path):
    h,w=2160,3840
    s= compensated
    mid_h = s.shape[0] // 2 - h // 2
    mid_w = s.shape[1] // 2 - w // 2
    s = s[mid_h:mid_h+h, mid_w:mid_w+w]
    g= current
    mid_h = g.shape[0] // 2 - h // 2
    mid_w = g.shape[1] // 2 - w // 2
    g = g[mid_h:mid_h+h, mid_w:mid_w+w]
    f = open(so_txt_path, 'r')

    mask = []
    for line in f.readlines():
        mask.append(int(line.strip('\n')))
    f.close()
    # h=540, w=960
    mask = np.array(mask).astype(bool)
    
    assert np.sum(mask) == 13000, 'The number of selected blocks should be 13000'

    s = s.reshape(h // 16, 16, w// 16, 16).swapaxes(1, 2).reshape(-1, 16, 16)
    g = g.reshape(h // 16, 16, w// 16, 16).swapaxes(1, 2).reshape(-1, 16, 16)

    s = s[mask]
    g = g[mask]
    assert not (s == g).all(), "The prediction should not be the same as the ground truth"

    mse = np.sum((s - g) ** 2) / s.size
    psnr = 10 * np.log10(255**2 / mse)

    return psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-s', '--so_path', type=str, required=True, help="Path to the source image and txt files")
    # parser.add_argument('-g', '--gt_path', type=str, required=True, help="Path to the ground truth image files")
    # parser.add_argument('-i', '--image_path', type=str, required=True, help="Index of the frame to evaluate")
    parser.add_argument('-f', '--frame_index', type=int, required=True, help="Index of the frame to evaluate")
    args = parser.parse_args()
    frame_index= args.frame_index
    so_path = f"./results/DaylightRoad2_27.yuv/sel_map/{frame_index:03d}.txt"
    gt_path = f"./results/DaylightRoad2_27.yuv/frames/{frame_index:03d}.png"
    frame_path= f'./results/DaylightRoad2_27.yuv/compensated/{frame_index:03d}.png'
    print(frame_path)
   
    if frame_index in [0, 32, 64, 96, 128]:
        print("Frame index cannot be 0, 32, 64, 96, or 128.")

    so_img_path = frame_path
    so_txt_path = so_path
    gt_img_path = gt_path

    so_img = cv2.imread(frame_path)
    gt_img = cv2.imread(gt_path)

    psnr = calculate_psnr_for_frame(so_img, gt_img, so_path)

    print(f'PSNR for frame {frame_index:03d}: {psnr:.5f}')

