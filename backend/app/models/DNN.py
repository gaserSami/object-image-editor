import cv2
import torch
import numpy as np
from PIL import Image

def norm_img(np_img):
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    np_img = np.transpose(np_img, (2, 0, 1))
    np_img = np_img.astype("float32") / 255
    return np_img

def pad_img_to_modulo(img, mod):
    """Pad image to be divisible by mod"""
    if len(img.shape) == 2:
        h, w = img.shape
        out_h = ((h + mod - 1) // mod) * mod
        out_w = ((w + mod - 1) // mod) * mod
        return np.pad(img, ((0, out_h - h), (0, out_w - w)), mode='reflect')
    else:
        c, h, w = img.shape
        out_h = ((h + mod - 1) // mod) * mod
        out_w = ((w + mod - 1) // mod) * mod
        return np.pad(img, ((0, 0), (0, out_h - h), (0, out_w - w)), mode='reflect')

def process_image(image, mask, model, device):
    """
    image: numpy array RGB image
    mask: numpy array mask where 255 is the area to inpaint
    """
    # Convert BGR to RGB if image was loaded with cv2.imread()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert image to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Store original dimensions
    original_height, original_width = image.shape[:2]
    
    # Normalize images
    image = norm_img(image)
    mask = norm_img(mask)
    
    # Pad both image and mask to be divisible by 8
    image = pad_img_to_modulo(image, 8)
    mask = pad_img_to_modulo(mask, 8)
    
    # Convert to torch tensors
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask).unsqueeze(0).to(device)
    
    # Process with model
    with torch.no_grad():
        result = model(image, mask)
    
    # Convert result back to numpy
    result = result[0].permute(1, 2, 0).detach().cpu().numpy()
    
    # Crop back to original size
    result = result[:original_height, :original_width, :]
    
    # Convert to uint8
    result = np.clip(result * 255, 0, 255).astype("uint8")
    
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)