# main.py
import os
import numpy as np
from PIL import Image
from yuv2png import convert
import cv2

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

def affine_motion_estimation(current_block, reference_frame, block_x, block_y, block_size, search_range):
    best_cost = float('inf')
    best_params = None
    
    h, w = current_block.shape
    current_points = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], dtype=np.float32)
    
    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            ref_points = np.array([[block_x+dx, block_y+dy], 
                                   [block_x+dx+w-1, block_y+dy], 
                                   [block_x+dx, block_y+dy+h-1], 
                                   [block_x+dx+w-1, block_y+dy+h-1]], dtype=np.float32)
            M = cv2.getAffineTransform(current_points[:3], ref_points[:3])
            transformed_block = cv2.warpAffine(reference_frame, M, (w, h))
            cost = np.sum(np.abs(current_block - transformed_block))
            if cost < best_cost:
                best_cost = cost
                best_params = M
    
    return best_params

def perspective_motion_estimation(current_block, reference_frame, block_x, block_y, block_size, search_range):
    best_cost = float('inf')
    best_params = None
    
    h, w = current_block.shape
    current_points = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]], dtype=np.float32)
    
    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            ref_points = np.array([[block_x+dx, block_y+dy], 
                                   [block_x+dx+w-1, block_y+dy], 
                                   [block_x+dx, block_y+dy+h-1], 
                                   [block_x+dx+w-1, block_y+dy+h-1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(current_points, ref_points)
            transformed_block = cv2.warpPerspective(reference_frame, M, (w, h))
            cost = np.sum(np.abs(current_block - transformed_block))
            if cost < best_cost:
                best_cost = cost
                best_params = M
    
    return best_params

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
                
                # Perform translational motion estimation
                vector = motion_estimation(current_block, reference_frame, x, y, block_size, search_range)
                
                # Perform affine motion estimation
                affine_params = affine_motion_estimation(current_block, reference_frame, x, y, block_size, search_range)
                
                # Perform perspective motion estimation
                perspective_params = perspective_motion_estimation(current_block, reference_frame, x, y, block_size, search_range)
                
                # Choose the best motion model based on cost
                # This is a simplified cost comparison. You can improve it based on your criteria.
                if affine_params is not None and perspective_params is not None:
                    affine_cost = np.sum(np.abs(current_block - cv2.warpAffine(reference_frame, affine_params, (block_size, block_size))))
                    perspective_cost = np.sum(np.abs(current_block - cv2.warpPerspective(reference_frame, perspective_params, (block_size, block_size))))
                    
                    if affine_cost < perspective_cost:
                        frame_motion_vectors.append(('affine', affine_params))
                        transformed_block = cv2.warpAffine(reference_frame, affine_params, (block_size, block_size))
                    else:
                        frame_motion_vectors.append(('perspective', perspective_params))
                        transformed_block = cv2.warpPerspective(reference_frame, perspective_params, (block_size, block_size))
                else:
                    frame_motion_vectors.append(('translational', vector))
                    ref_x = x + vector[0]
                    ref_y = y + vector[1]
                    transformed_block = reference_frame[ref_y:ref_y+block_size, ref_x:ref_x+block_size]
                
                predicted_frame[y:y+block_size, x:x+block_size] = transformed_block
                
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
