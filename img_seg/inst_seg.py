import torch
from torchvision.io.image import read_image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_pil_image
import numpy as np
from PIL import Image, ImageDraw, ImageOps

# Load the image
img = read_image("./image/037.png")

# Initialize model with the best available weights
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights)
model.eval()

# Initialize the inference transforms
preprocess = weights.transforms()

# Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Use the model and get the prediction
with torch.no_grad():
    prediction = model(batch)

# Process the prediction to extract masks and bounding boxes
masks = prediction[0]['masks']  # Shape: [N, 1, H, W]
labels = prediction[0]['labels']
scores = prediction[0]['scores']
boxes = prediction[0]['boxes']

# Threshold to filter out low-confidence predictions
score_threshold = 0.5
keep = scores >= score_threshold

masks = masks[keep]
labels = labels[keep]
boxes = boxes[keep]

# Create a blank RGBA image for the instance segmentation map
segmentation_image = Image.new("RGBA", (img.shape[2], img.shape[1]), (0, 0, 0, 0))

# Create a color palette for visualization
colors = np.random.randint(0, 255, (len(labels), 3), dtype=np.uint8)

for mask, box, color in zip(masks, boxes, colors):
    mask = mask.squeeze(0).cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255  # Convert mask to binary and scale to 255

    # Create an RGB mask
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb_mask[mask == 255] = color

    # Convert the RGB mask to a PIL image and add alpha channel
    pil_mask = Image.fromarray(rgb_mask).convert("RGBA")
    alpha_mask = Image.fromarray(mask).convert("L")
    pil_mask.putalpha(alpha_mask)

    # Paste the mask onto the segmentation image
    segmentation_image = Image.alpha_composite(segmentation_image, pil_mask)

# Save the instance segmentation image
output_path = "./image/037_inst_seg.png"
segmentation_image.save(output_path)

# Optionally, display the image
segmentation_image.show()
