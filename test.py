def hierarchical_b_structure():
    # The frame ranges and their processing orders as described in the image
    frame_ranges = [(1, 31), (33, 63), (65, 95), (97, 127)]
    predefined_skipped_frames = [0, 32, 64, 96, 128]

    def add_references(start, end, structure):
        # Base case: if only one frame, it has no B-frame references
        if start == end:
            structure.append({'curr': start, 'l': start-1, 'r': start+1, 't': 'B'})
            return
        # Find the middle frame
        mid = (start + end) // 2
        
        # Add the middle frame referencing its range boundaries
        left_ref = start-1
        right_ref = end+1
        
        structure.append({'curr': mid, 'l': left_ref, 'r': right_ref})
        
        # Recursively add references for left and right sub-ranges
        add_references(start, mid-1, structure)
        add_references(mid+1, end, structure)

    hierarchical_structure = []
    
    for start, end in frame_ranges:
        add_references(start, end, hierarchical_structure)
    for idx in predefined_skipped_frames:
        hierarchical_structure.append({'curr': idx, 'l': None, 'r': None})
    hierarchical_structure.sort(key=lambda x: x['curr'])
    return hierarchical_structure

# Example usage
hierarchical_structure = hierarchical_b_structure()
for frame in hierarchical_structure:
    print(frame)