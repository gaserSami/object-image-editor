from app.utils.image_utils import ImageUtils
import numpy as np
import cv2 as cv
from app.models.seam_carving import remove, add_seams, remove_seams

class SeamCarverService:
    @staticmethod
    def resize_with_mask(image_data, new_height, new_width, protect_mask):
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
        
        try:
            # Calculate size differences
            current_height, current_width = image.shape[:2]
            dy = int(new_height) - current_height 
            dx = int(new_width) - current_width

            output = image.copy()
            
            # Handle width change
            if dx != 0:
                if dx < 0:
                    # Remove seams to decrease width
                    output, _ = remove_seams(output, abs(dx), protect_mask)
                else:
                    # Add seams to increase width
                    output, _ = add_seams(output, dx, protect_mask)
                    
            # Handle height change by rotating image 90 degrees
            if dy != 0:
                output = np.rot90(output)
                if protect_mask is not None:
                    protect_mask = np.rot90(protect_mask)
                
                if dy < 0:
                    # Remove seams to decrease height
                    output, _ = remove_seams(output, abs(dy), protect_mask)
                else:
                    # Add seams to increase height
                    output, _ = add_seams(output, dy, protect_mask)
                    
                # Rotate image back to original orientation
                output = np.rot90(output, -1)
            
            return output.astype(np.uint8)
            
        except ValueError as e:
            print(e)
            return image

    @staticmethod
    def remove_object(image_data, object_mask, protect_mask):
        """
        Remove object from image using seam carving.
        
        Parameters:
        image_data (str): Base64 encoded image data.
        object_mask (str): Base64 encoded mask of the object to be removed.
        protect_mask (str): Base64 encoded protective mask to preserve certain areas.
        
        Returns:
        np.ndarray: Image with the object removed.
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
            
        # Perform object removal
        output, _ = remove(image, remove_mask=object_mask, protect_mask=protect_mask)
        
        return output.astype(np.uint8)