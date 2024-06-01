import numpy as np

def encode_residuals(residuals):
    # Perform quantization or any other compression technique here
    encoded_residuals = residuals.astype(np.int16)  # Example: Converting to int16
    return encoded_residuals

def encode_motion_vectors(motion_vectors):
    # Perform entropy coding or any other compression technique here
    encoded_motion_vectors = np.array(motion_vectors)  # Example: No compression, just convert to numpy array
    return encoded_motion_vectors

def encode(frames, motion_vectors):
    encoded_frames = []
    encoded_motion_vectors = encode_motion_vectors(motion_vectors)
    
    for i, frame in enumerate(frames):
        residual = frame - motion_vectors[i]
        encoded_residual = encode_residuals(residual)
        encoded_frames.append(encoded_residual)
    
    return encoded_frames, encoded_motion_vectors

if __name__ == "__main__":
    # Example usage
    frames = [np.array([[1, 2], [3, 4]]), np.array([[2, 3], [4, 5]]), np.array([[3, 4], [5, 6]])]
    motion_vectors = [(0, 0), (1, 1), (1, 1)]

    encoded_frames, encoded_motion_vectors = encode(frames, motion_vectors)
    print("Encoded Frames:")
    for encoded_frame in encoded_frames:
        print(encoded_frame)
    print("Encoded Motion Vectors:", encoded_motion_vectors)
