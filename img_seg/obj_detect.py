from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import argparse
import os
import math
import numpy as np
from PIL import Image
from PIL import ImageColor

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--so_path', type=str)
parser.add_argument('-g', '--gt_path', type=str)
parser.add_argument('-m', '--map_path', type=str)
args = parser.parse_args()

so_path = args.so_path
#mp_path = args.map_path

image_name = ['%03d.png'% i for i in range(129) if i not in [0, 32, 64, 96, 128]]
#txt_name   = ['s_%03d.txt'% i for i in range(129) if i not in [0, 32, 64, 96, 128]]
so_img_paths = [os.path.join(so_path,name) for name in image_name]
#so_txt_paths = [os.path.join(mp_path,name) for name in txt_name]

#for so_img_path, so_txt_path in zip(so_img_paths, so_txt_paths):
for so_img_path in zip(so_img_paths):       
    print('check image... ', so_img_path)

    #f = open(so_txt_path, 'r')
    so_img_path = so_img_path[0]
    img = read_image(so_img_path)

    # Step 1: Initialize model with the best available weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = [preprocess(img)]

    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                            labels=labels,
                            colors="red",
                            width=4, font_size=30)
    im = to_pil_image(box.detach())
    output_path = so_img_path.replace('.png','_detect.png')
    im.save(output_path)

    mask = np.zeros((img.shape[1], img.shape[2]))  # Assuming img is a CxHxW tensor
    object_id = 1  # Start the object IDs from 1
    for box in prediction["boxes"]:
        x1, y1, x2, y2 = box.int()
        mask[y1:y2, x1:x2] = object_id  # Set the pixels inside the box to the object ID
        object_id += 1  # Increment the object ID for the next object
        
    # Here we use the 'tab20' colormap which provides 20 distinct colors
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'olive', 'teal', 'aqua', 'fuchsia', 'gray', 'silver', 'black', 'white']

    # Create a 16x16 block mask map
    block_size = 16
    h_blocks = math.ceil(img.shape[1] / block_size)
    w_blocks = math.ceil(img.shape[2] / block_size)
    block_mask = np.zeros((h_blocks, w_blocks))
    block_image= np.zeros((h_blocks, w_blocks, 3)) 
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = mask[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            unique_ids = np.unique(block)
            # If the block contains more than one object, set the value to -1
            block_mask[i, j] = unique_ids[0] if len(unique_ids) == 1 else -1
            block_id = int(block_mask[i, j])
            block_color = ImageColor.getrgb(colors[block_id % len(colors)])  # Get the color corresponding to the block ID
            block_image[i, j] = block_color  # Set the block color
  # Convert color to 8-bit values

    # Save the block mask map as a .txt file
    np.savetxt(so_img_path.replace('.png', '_block_mask.txt'), block_mask, fmt='%d')
    # Create a colormap that maps each unique object ID to a different color


    block_mask_img = Image.fromarray((block_image).astype(np.uint8))  # Convert to 8-bit pixel values
    block_mask_img.save(so_img_path.replace('.png', '_block_mask.png'))