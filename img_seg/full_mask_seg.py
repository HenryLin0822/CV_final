import torch
from torchvision.io.image import read_image
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from torchvision.transforms.functional import to_pil_image
import numpy as np
from PIL import Image

# Function to downsample the predicted classes using 16x16 blocks
def downsample_segmentation(segmentation, block_size=16):
    h, w = segmentation.shape
    h_blocks = h // block_size
    w_blocks = w // block_size
    downsampled = np.zeros((h_blocks, w_blocks), dtype=int)
    print(segmentation.shape)
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = segmentation[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            mode_value = np.bincount(block.flatten()).argmax()
            downsampled[i, j] = mode_value
    print(downsampled.shape)
    return downsampled

def upscale_segmentation(segmentation, original_size):
    pil_img = Image.fromarray(segmentation.astype(np.uint8), mode='L')
    upscaled_img = pil_img.resize(original_size, resample=Image.NEAREST)
    return np.array(upscaled_img)

img = read_image("./image/037.png")
print(img.shape)
# Step 1: Initialize model with the best available weights
weights = DeepLabV3_ResNet101_Weights.DEFAULT
model = deeplabv3_resnet101(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)
print(batch.shape)
# Step 4: Use the model and visualize the prediction
with torch.no_grad():
    prediction = model(batch)["out"]
    normalized_masks = prediction.softmax(dim=1)

# Step 5: Get the class with the highest probability for each pixel
_, predicted_classes = normalized_masks.max(dim=1)

# Convert tensor to a PIL image
predicted_classes = predicted_classes.squeeze(0).cpu().numpy()
predict_upscale = upscale_segmentation(predicted_classes,[2160,3840])
downsampled_classes = downsample_segmentation(predict_upscale)

# Output downsampled_classes to map.txt
np.savetxt("./image/map.txt", downsampled_classes, fmt='%d')
# Map class indices to colors (using a simple color map for visualization)
def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
                             (128, 0, 0),  # 1=aeroplane
                             (0, 128, 0),  # 2=bicycle
                             (128, 128, 0),  # 3=bird
                             (0, 0, 128),  # 4=boat
                             (128, 0, 128),  # 5=bottle
                             (0, 128, 128),  # 6=bus
                             (128, 128, 128),  # 7=car
                             (64, 0, 0),  # 8=cat
                             (192, 0, 0),  # 9=chair
                             (64, 128, 0),  # 10=cow
                             (192, 128, 0),  # 11=dining table
                             (64, 0, 128),  # 12=dog
                             (192, 0, 128),  # 13=horse
                             (64, 128, 128),  # 14=motorbike
                             (192, 128, 128),  # 15=person
                             (0, 64, 0),  # 16=potted plant
                             (128, 64, 0),  # 17=sheep
                             (0, 192, 0),  # 18=sofa
                             (128, 192, 0),  # 19=train
                             (0, 64, 128)])  # 20=tv/monitor

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    
    rgb = np.stack([r, g, b], axis=2)
    return rgb

# Decode the segmentation map
segmentation_map = decode_segmap(predicted_classes)

# Convert to PIL image and display
segmentation_image = Image.fromarray(segmentation_map)
output_path = "./image/000_seg.png"
segmentation_image.save(output_path)

# Optionally, display the image
segmentation_image.show()

