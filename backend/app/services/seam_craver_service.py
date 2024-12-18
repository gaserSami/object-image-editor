from app.utils.image_utils import ImageUtils
import numpy as np
import cv2 as cv
from app.models.seam_carving import remove, add_seams, remove_seams
import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt

class SeamCarverService:
    @staticmethod
    def resize_with_mask(image_data, new_height, new_width, protect_mask, forward=True):
        """
        Resize image using seam carving with optional protective mask.
        
        Parameters:
        image_data (str): Base64 encoded image data.
        new_height (int): Desired height of the resized image.
        new_width (int): Desired width of the resized image.
        protect_mask (str): Base64 encoded protective mask to preserve certain areas.
        
        Returns:
        np.ndarray: Resized image.
        """
        # Decode base64 image data
        image = ImageUtils.decode_base64_image(image_data)
        
        if protect_mask is not None:
            # Decode and process the protective mask
            protect_mask = ImageUtils.decode_base64_image(protect_mask)
            protect_mask = cv.cvtColor(protect_mask, cv.COLOR_BGR2GRAY)
            protect_mask = protect_mask.astype(np.float64)
        
        # Calculate size differences
        current_height, current_width = image.shape[:2]
        dy = int(new_height) - current_height 
        dx = int(new_width) - current_width

        output = image.copy()
        
        # Handle width change
        if dx != 0:
            if dx < 0:
                # Remove seams to decrease width
                output, _, _ = remove_seams(output, abs(dx), protect_mask, forward=forward)
            else:
                # Add seams to increase width
                output, _ = add_seams(output, dx, protect_mask, forward=forward)
                
        # Handle height change by rotating image 90 degrees
        if dy != 0:
            output = np.rot90(output)
            if protect_mask is not None:
                protect_mask = np.rot90(protect_mask)
            
            if dy < 0:
                # Remove seams to decrease height
                output, _, _ = remove_seams(output, abs(dy), protect_mask, forward=forward)
            else:
                # Add seams to increase height
                output, _ = add_seams(output, dy, protect_mask, forward=forward)
                
            # Rotate image back to original orientation
            output = np.rot90(output, -1)
        
        return output.astype(np.uint8)

    @staticmethod
    def remove_object(image_data, object_mask, protect_mask, forward=True, direction="auto"):
        """
        Remove object from image using seam carving.
        
        Parameters:
        image_data (str): Base64 encoded image data.
        object_mask (str): Base64 encoded mask of the object to be removed.
        protect_mask (str): Base64 encoded protective mask to preserve certain areas.
        forward (bool): Whether to use forward energy (default: True)
        direction (str): Optional direction for seam removal ('vertical', 'horizontal', or None for auto)
        
        Returns:
        tuple: (np.ndarray, str) - (Image with object removed, Path to generated GIF)
        """
        # Decode base64 image data
        image = ImageUtils.decode_base64_image(image_data)
        
        # Process object mask
        if object_mask is not None:
            object_mask = ImageUtils.decode_base64_image(object_mask)
            object_mask = cv.cvtColor(object_mask, cv.COLOR_BGR2GRAY)
            object_mask = np.where(object_mask > 0, 1, 0)
            
        # Process protect mask  
        if protect_mask is not None:
            protect_mask = ImageUtils.decode_base64_image(protect_mask)
            protect_mask = cv.cvtColor(protect_mask, cv.COLOR_BGR2GRAY)
            protect_mask = np.where(protect_mask > 0, 1, 0)
            protect_mask = protect_mask.astype(np.float64)
        
        intermediate_images = [] # to be global
        
        # Perform object removal
        try:
            output, _, intermediate_images = remove(
                image, 
                remove_mask=object_mask, 
                protect_mask=protect_mask, 
                forward=forward,
                direction=direction
            )
        except Exception as e:
            gif_path = make_gif(intermediate_images=intermediate_images, image=image)
            print(gif_path)
            print(e)
            return image, None
        
        gif_path = make_gif(intermediate_images=intermediate_images, image=image)
        print(gif_path)
        
        return output.astype(np.uint8)
    
def make_gif(intermediate_images, image):
    # Create tmp directory if it doesn't exist
    os.makedirs('tmp', exist_ok=True)
    
    # Generate unique filename
    gif_path = os.path.join('tmp', f'seam_carving.gif')
    
    # Convert images to uint8 and pad smaller images to match the original size
    uint8_images = []
    original_shape = image.shape
    channels = 1 if len(original_shape) == 2 else original_shape[2]
    
    for img in intermediate_images:
        # Convert to float64 for normalization operations
        img = img.astype(np.float64)
        
        # Handle normalized values (0-1 range)
        if img.max() <= 1.0:
            img = img * 255
            
        # Handle both 3-channel and 1-channel images
        if len(img.shape) != len(original_shape):
            if len(img.shape) == 2 and channels == 3:
                img = np.stack((img,) * 3, axis=-1)
            elif len(img.shape) == 3 and channels == 1:
                img = img[:,:,0]
                
        # Create padded image with correct dimensions
        if channels == 1:
            padded_img = np.zeros(original_shape, dtype=np.uint8)
        else:
            padded_img = np.zeros((original_shape[0], original_shape[1], channels), dtype=np.uint8)
            
        h, w = img.shape[:2]
        if channels == 1:
            padded_img[:h, :w] = np.clip(img, 0, 255).astype(np.uint8)
        else:
            padded_img[:h, :w, :] = np.clip(img, 0, 255).astype(np.uint8)
            
        uint8_images.append(padded_img)
        
    imageio.mimsave(gif_path, uint8_images, duration=0.1)  # 0.1 seconds per frame
    return gif_path
