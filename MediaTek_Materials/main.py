# main.py
import os
import numpy as np
from PIL import Image
from yuv2png import convert

def load_frames_from_directory(directory, seq_len):
    frames = []
    for frame_num in range(seq_len):
        frame_path = os.path.join(directory, f'{frame_num:03d}.png')
        if os.path.exists(frame_path):
            frame = Image.open(frame_path).convert('L')
            frames.append(np.array(frame))
    return frames

def motion_estimation(current_block, reference_frame, block_x, block_y, block_size, search_range):
    best_cost = float('inf')
    best_vector = (0, 0)
    
    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            ref_x = block_x + dx
            ref_y = block_y + dy
            if 0 <= ref_x < reference_frame.shape[1] - block_size and 0 <= ref_y < reference_frame.shape[0] - block_size:
                ref_block = reference_frame[ref_y:ref_y+block_size, ref_x:ref_x+block_size]
                cost = np.sum(np.abs(current_block - ref_block))
                if cost < best_cost:
                    best_cost = cost
                    best_vector = (dx, dy)
    
    return best_vector

def motion_compensation(frames, block_size, search_range):
    motion_vectors = []
    compensated_frames = [frames[0]]  # The first frame remains unchanged

    for i in range(1, len(frames)):
        reference_frame = frames[i-1]
        current_frame = frames[i]
        h, w = current_frame.shape
        predicted_frame = np.zeros_like(current_frame)
        frame_motion_vectors = []

        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                current_block = current_frame[y:y+block_size, x:x+block_size]
                vector = motion_estimation(current_block, reference_frame, x, y, block_size, search_range)
                frame_motion_vectors.append(vector)
                ref_x = x + vector[0]
                ref_y = y + vector[1]
                predicted_frame[y:y+block_size, x:x+block_size] = reference_frame[ref_y:ref_y+block_size, ref_x:ref_x+block_size]

        motion_vectors.append(frame_motion_vectors)
        compensated_frames.append(predicted_frame)
    
    return compensated_frames, motion_vectors

def save_compensated_frames(frames, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        img.save(os.path.join(output_dir, f'compensated_{i:03d}.png'))
        print(f"Saved compensated frame compensated_{i:03d}.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Perform motion compensation on YUV420 video.')
    parser.add_argument('-y', '--yuv_file', required=True, help='Path to the YUV file.')
    parser.add_argument('-o', '--output_dir', required=True, help='Directory to save compensated frames.')
    parser.add_argument('-w', '--width', type=int, required=True, help='Width of the frames.')
    parser.add_argument('-h', '--height', type=int, required=True, help='Height of the frames.')
    parser.add_argument('-s', '--seq_len', type=int, required=True, help='Number of frames in the sequence.')
    parser.add_argument('-b', '--block_size', type=int, default=16, help='Block size for motion estimation.')
    parser.add_argument('-r', '--search_range', type=int, default=4, help='Search range for motion estimation.')

    args = parser.parse_args()

    # Convert YUV to PNG frames
    intermediate_dir = os.path.join(args.output_dir, 'intermediate_frames')
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir)
    
    convert(args.yuv_file, intermediate_dir, args.width, args.height, args.seq_len)

    # Load frames from the output directory
    frames = load_frames_from_directory(intermediate_dir, args.seq_len)

    # Perform motion compensation
    compensated_frames, motion_vectors = motion_compensation(frames, args.block_size, args.search_range)

    # Save compensated frames
    save_compensated_frames(compensated_frames, args.output_dir)
