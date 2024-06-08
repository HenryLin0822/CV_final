from utils import get_video_frames, draw_motion_field, PSNR, read_frames_from_directory, hierarchical_b_structure, convert_png_to_mp4, read_object_map
from eval_one_img import calculate_psnr_for_frame
import motion as motion
from json import dump
import numpy as np
import torch
import argparse
import shutil
import cv2
import os
from concurrent.futures import ProcessPoolExecutor, as_completed



FRAME_DISTANCE = 1
def calculate_block_mse(current_frame, ground_truth_frame):
    block_size = 16
    mse_scores = []

    height, width = current_frame.shape[:2]

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block_current = current_frame[y:y+block_size, x:x+block_size]
            block_gt = ground_truth_frame[y:y+block_size, x:x+block_size]

            mse = np.mean((block_current - block_gt) ** 2)
            mse_scores.append((mse, (y // block_size, x // block_size)))

    mse_scores.sort(key=lambda x: x[0])  # Sort by MSE, lowest first
    return mse_scores

def select_top_blocks(mse_scores, num_blocks=13000):
    selected_blocks = []

    for _, (block_y, block_x) in mse_scores[:num_blocks]:
        selected_blocks.append((block_y, block_x))

    return selected_blocks

def save_block_mask(frame_index, selected_blocks, save_dir, frame_height, frame_width):
    block_size = 16
    mask_height = frame_height // block_size
    mask_width = frame_width // block_size
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

    for block_y, block_x in selected_blocks:
        if block_y < mask_height and block_x < mask_width:
            mask[block_y, block_x] = 1

    # Save mask as PNG
    cv2.imwrite(os.path.join(save_dir, f"{int(frame_index):03d}.png"), mask * 255)  # Multiply by 255 to visualize

    # Save mask as TXT
    
    with open(os.path.join(save_dir, f"s_{int(frame_index):03d}.txt"), "a") as f:
        for row in mask:
            for value in row:
                f.write(str(value) + "\n")

def process_frame(hevc, frames, obj_map, save_path, psnr_dict):
    # Use MPS (Metal Performance Shaders) if available
    device = torch.device("cpu")
    
    idx = hevc['curr']
    print(idx, hevc['l'], hevc['r'])
    

    
    if hevc['t'] == 'I':
        reference_l = frames[idx]
        reference_r = frames[idx]
        current = frames[idx]
        compensated = frames[idx]
    else:
        reference_l = frames[hevc['l']]
        reference_l_obj = obj_map[hevc['l']]
        reference_r = frames[hevc['r']]
        reference_r_obj = obj_map[hevc['r']]
        current = frames[idx]
        # if hevc['r']-hevc['l']>=16:
        #     d = (hevc['r']-hevc['l'])/2
        #     params_l=[]
        #     model_motion_field_l_list = []
        #     params_r=[]
        #     model_motion_field_r_list = []
        #     for i in range(int(np.log2(d))):
        #         reference_l_first = frames_tensor[hevc['l']+2*i]
        #         reference_l_first_obj = obj_map[hevc['l']+2*i]
        #         reference_l_sec = frames_tensor[hevc['l']+2*(i+1)]
        #         #reference_l_sec_obj = obj_map[hevc['l']+2*i]
        #         params_l.append(motion.global_motion_estimation(reference_l_first , reference_l_sec , reference_l_first_obj))
        #         model_motion_field_l_list.append(motion.get_motion_field_affine(
        #         (int(current.shape[0] / motion.BBME_BLOCK_SIZE), int(current.shape[1] / motion.BBME_BLOCK_SIZE), 2), parameters=(params_l[i])
        #     ))
        #     for i in range(int(np.log2(d))):
        #         reference_r_first = frames_tensor[hevc['r']-2*i]
        #         reference_r_first_obj = obj_map[hevc['r']-2*i]
        #         reference_r_sec = frames_tensor[hevc['r']-2*(i+1)]
        #         #reference_l_sec_obj = obj_map[hevc['l']+2*i]
        #         params_r.append(motion.global_motion_estimation(reference_r_first , reference_r_sec , reference_r_first_obj))
        #         model_motion_field_r_list.append(motion.get_motion_field_affine(
        #         (int(current.shape[0] / motion.BBME_BLOCK_SIZE), int(current.shape[1] / motion.BBME_BLOCK_SIZE), 2), parameters=(params_r[i])
        #     ))
        #     model_motion_field_r=sum(model_motion_field_r_list)
        #     model_motion_field_l=sum(model_motion_field_r_list)
        # else:    
        params_l = motion.global_motion_estimation(reference_l, current, reference_l_obj)
        params_r = motion.global_motion_estimation(reference_r, current, reference_r_obj)


        model_motion_field_l = motion.get_motion_field_affine(
            (int(current.shape[0] / motion.BBME_BLOCK_SIZE), int(current.shape[1] / motion.BBME_BLOCK_SIZE), 2), parameters=(params_l)
        )
        model_motion_field_r = motion.get_motion_field_affine(
            (int(current.shape[0] / motion.BBME_BLOCK_SIZE), int(current.shape[1] / motion.BBME_BLOCK_SIZE), 2), parameters=(params_r)
        )
        # if idx % 2 == 0:
        #     v1 = np.sqrt(np.mean(model_motion_field_l**2))
        #     v2 = np.sqrt(np.mean(model_motion_field_r**2))
        #     t = (hevc['r'] - hevc['l']) / 6
        #     a = (v2 - v1) / t
        #     s = v1 * t + 0.5 * a * t**2
        #     s_d = v1 * t / 2 + 0.5 * a * t**2 / 4
        #     l_weight = 1 - s_d / s
        #     r_weight = s_d / s
        # else:
        

        # Compensate camera motion on reference frame
        #compensated_l = motion.compensate_frame(reference_l , model_motion_field_l)
        #compensated_r = motion.compensate_frame(reference_r , model_motion_field_r)
        if idx%2==0:
            compensated = motion.bi_compensate_frame(reference_l , model_motion_field_l, reference_r , model_motion_field_r)
        else:
            compensated_l = motion.compensate_frame(reference_l , model_motion_field_l)
            compensated_r = motion.compensate_frame(reference_r , model_motion_field_r)
            compensated = 0.5*compensated_l+0.5*compensated_r
 
        idx_name = str(idx).zfill(3)
        mse_scores = calculate_block_mse(current , compensated)
        selected_blocks = select_top_blocks(mse_scores, 13000)
        save_block_mask(idx_name, selected_blocks, os.path.join(save_path, "sel_map"), current.shape[0], current.shape[1])
        
        diff_curr_prev = np.abs(current .astype("int") - reference_l .astype("int")).astype("uint8")
        diff_curr_comp = np.abs(current .astype("int") - compensated.astype("int")).astype("uint8")
        cv2.imwrite(os.path.join(save_path, "curr_prev_diff", "") + str(idx).zfill(3) + ".png", diff_curr_prev)
        cv2.imwrite(os.path.join(save_path, "curr_comp_diff", "") + str(idx).zfill(3) + ".png", diff_curr_comp)

        psnr = calculate_psnr_for_frame(compensated, current , os.path.join(save_path, "sel_map", f"s_{int(idx):03d}.txt"))
        psnr_dict[idx_name] = str(psnr)
        with open(save_path + "psnr_records.txt", "a") as outfile:
           outfile.write(str(idx_name)+": "+str(psnr) + " ") 

    idx_name = str(idx)
    cv2.imwrite(os.path.join(save_path, "frames", "") + str(idx).zfill(3) + ".png", current )
    cv2.imwrite(os.path.join(save_path, "compensated", "") + str(idx).zfill(3) + ".png", compensated)



def parallel_process_frames(hevc_b, frames, obj_map, save_path, psnr_dict):
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_frame, hevc, frames, obj_map, save_path, psnr_dict) for hevc in hevc_b]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing frame: {e}")


def main(args):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("mps is available")
    else:
        device = torch.device("cpu")

    video = args.path
    if args.fd is not None:
        FRAME_DISTANCE = int(args.fd)
    video_path = os.path.join("resources", "videos", video)
    results_path = os.path.join("results", "")
    save_path = os.path.join(results_path, video.replace(".mp4", ""), "")
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    os.mkdir(save_path)
    os.mkdir(os.path.join(save_path, "frames", ""))
    os.mkdir(os.path.join(save_path, "compensated", ""))
    os.mkdir(os.path.join(save_path, "curr_prev_diff", ""))
    os.mkdir(os.path.join(save_path, "model_motion_field", ""))
    os.mkdir(os.path.join(save_path, "curr_comp_diff", ""))
    os.mkdir(os.path.join(save_path, "sel_map", ""))

    psnr_dict = {}
    frames=read_frames_from_directory("./resources/frame", h=2160, w=3840)
    obj_map=read_object_map()
    hevc_b = hierarchical_b_structure()
    try:
        print("frame shape: {}".format(frames[0].shape))
    except:
        raise Exception("Error reading video file: check the name of the video!")

        # # save differences
    parallel_process_frames(hevc_b, frames, obj_map, save_path, psnr_dict)
          
    # convert_png_to_mp4(save_path+'/compensated', 'output.mp4')
    # convert_png_to_mp4(save_path+'/frames', 'gt_output.mp4')
     

if __name__ == "__main__":
    """Once set the video path and save path creates a lot of data for your report. Namely, it saves frames, compensated frames, frame differences and estimations of global motion.
    """
    parser = argparse.ArgumentParser(
        description="Launches GME and yields results"
    )
    parser.add_argument(
        "-v",
        "--video-name",
        dest="path",
        type=str,
        required=True,
        help="name of the video to analyze (no ext)",
    )
    parser.add_argument(
        "-f",
        "--frame-distance",
        dest="fd",
        type=str,
        required=False,
        help="frame displacement",
    )
    args = parser.parse_args()

    main(args)



    