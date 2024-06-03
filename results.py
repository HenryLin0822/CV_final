from utils import get_video_frames, draw_motion_field, PSNR, read_frames_from_directory, hierarchical_b_structure, convert_png_to_mp4
import motion as motion
from json import dump
import numpy as np
import argparse
import shutil
import cv2
import os



FRAME_DISTANCE = 1
def calculate_block_mse(current_frame, ground_truth_frame):
    block_size = 16
    mse_scores = []

    height, width = current_frame.shape[:2]
    print()
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
    with open(os.path.join(save_dir, f"s_{int(frame_index):03d}.txt"), "w") as f:
        for row in mask:
            for value in row:
                f.write(str(value) + "\n")

def main(args):

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

    psnr_dict = {}
    frames=read_frames_from_directory("./resources/frame", h=2160, w=3840)
    hevc_b = hierarchical_b_structure()
    try:
        print("frame shape: {}".format(frames[0].shape))
    except:
        raise Exception("Error reading video file: check the name of the video!")
    for idx in range(len(frames)):
        j = (idx + 1) / len(frames)
        print("[%-20s] %d/%d frames" % ("=" * int(20 * j), idx, len(frames)))
        # get reference, current and compensated
        if hevc_b[idx]['t'] =='I':
           print(idx)
           reference_l = frames[idx]
           reference_r = frames[idx]
           current = frames[idx]
           compensated = frames[idx]
        else:
            reference_l = frames[hevc_b[idx]['l']]
            reference_r = frames[hevc_b[idx]['r']]
            current = frames[idx]

            params_l = motion.global_motion_estimation(current,reference_l)
            # print(params_l)
            params_r = motion.global_motion_estimation(current,reference_r)
            # print(params_r)
            print((params_l+params_r)/2)
            l_model_motion_field = motion.get_motion_field_affine(
                (int(current.shape[0] / motion.BBME_BLOCK_SIZE), int(current.shape[1] / motion.BBME_BLOCK_SIZE), 2), parameters=(params_l)
            )
            r_model_motion_field = motion.get_motion_field_affine((int(current.shape[0] / motion.BBME_BLOCK_SIZE), int(current.shape[1] / motion.BBME_BLOCK_SIZE), 2), parameters=(params_r))


            shape = (current.shape[0]//motion.BBME_BLOCK_SIZE, current.shape[1]//motion.BBME_BLOCK_SIZE)
            # compensate camera motion on reference frame
            compensated_l = motion.compensate_frame(reference_l, l_model_motion_field)
            compensated_r = motion.compensate_frame(reference_r, r_model_motion_field)
            compensated=(compensated_l+compensated_r)/2
            
            idx_name=str(idx).zfill(3)
            mse_scores=calculate_block_mse(current,compensated)
            selected_blocks=select_top_blocks(mse_scores,current.shape[0]//16 *current.shape[1] //16)
            save_block_mask(idx_name,selected_blocks,"./results/sel_map",current.shape[0],current.shape[1])
            diff_curr_prev = (
                np.absolute(current.astype("int") - reference_l.astype("int"))
            ).astype("uint8")
            diff_curr_comp = (
                np.absolute(current.astype("int") - compensated.astype("int"))
            ).astype("uint8")
            cv2.imwrite(
                os.path.join(save_path, "curr_prev_diff", "")
                + str(idx).zfill(3)
                + ".png",
                diff_curr_prev,
            )
            cv2.imwrite(
                os.path.join(save_path, "curr_comp_diff", "")
                + str(idx).zfill(3)
                + ".png",
                diff_curr_comp,
            )

            # save motion model motion field
            # draw = draw_motion_field(current, model_motion_field)
            # cv2.imwrite(
            #     os.path.join(save_path, "model_motion_field", "")
            #     + str(idx).zfill(3)
            #     + ".png",
            #     draw,
            # )


        idx_name = str(idx)
        # save frames
        cv2.imwrite(
            os.path.join(save_path, "frames", "")
            + str(idx).zfill(3)
            + ".png",
            current,
        )

        # save compensated
        cv2.imwrite(
            os.path.join(save_path, "compensated", "")
            + str(idx).zfill(3)
            + ".png",
            compensated,
        )

        # save differences
        

        ## compute PSNR
        psnr = PSNR(current, compensated)
        psnr_dict[idx_name] = str(psnr)
        with open(save_path + "psnr_records.json", "w") as outfile:
            dump(psnr_dict, outfile)
        
    # convert_png_to_mp4('./results/DaylightRoad2_27.yuv/compensated', 'output.mp4')
    # convert_png_to_mp4('./results/DaylightRoad2_27.yuv/frames', 'gt_output.mp4') 

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



    