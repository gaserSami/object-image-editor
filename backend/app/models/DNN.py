import cv2
import torch
import numpy as np

def _norm_img(np_img):
    """
    Normalize a numpy image array to a format suitable for model input.
    
    Parameters:
    np_img (numpy.ndarray): Input image array, can be 2D (grayscale) or 3D (color).
    
    Returns:
    numpy.ndarray: Normalized image array with shape (C, H, W) and float32 type.
    """
    # If the image is grayscale (2D), add a channel dimension
    if len(np_img.shape) == 2:
        np_img = np_img[:, :, np.newaxis]
    
    # Transpose the image to (C, H, W) format
    np_img = np.transpose(np_img, (2, 0, 1))
    
    # Normalize pixel values to the range [0, 1]
    np_img = np_img.astype("float32") / 255
    
    return np_img

def _pad_img_to_modulo(img, mod):
    """
    Pad an image to ensure its dimensions are divisible by a specified modulo.
    
    Parameters:
    img (numpy.ndarray): Input image array, can be 2D (grayscale) or 3D (color).
    mod (int): The modulo value to pad the image dimensions to.
    
    Returns:
    numpy.ndarray: Padded image array.
    """
    # Check if the image is grayscale (2D)
    if len(img.shape) == 2:
        h, w = img.shape
        # Calculate the new dimensions that are divisible by 'mod'
        out_h = ((h + mod - 1) // mod) * mod
        out_w = ((w + mod - 1) // mod) * mod
        # Pad the image using 'reflect' mode
        return np.pad(img, ((0, out_h - h), (0, out_w - w)), mode='reflect')
    else:
        # For color images (3D)
        c, h, w = img.shape
        # Calculate the new dimensions that are divisible by 'mod'
        out_h = ((h + mod - 1) // mod) * mod
        out_w = ((w + mod - 1) // mod) * mod
        # Pad the image using 'reflect' mode, keeping the channel dimension unchanged
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
    image = _norm_img(image)
    mask = _norm_img(mask)
    
    # Pad both image and mask to be divisible by 8
    image = _pad_img_to_modulo(image, 8)
    mask = _pad_img_to_modulo(mask, 8)
    
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