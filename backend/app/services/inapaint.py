from app.utils.image_utils import ImageUtils
from app.models.inpaint import Inpainter
import numpy as np
import cv2
import matplotlib.pyplot as plt

class InpaintingService:
    @staticmethod
    def inpaint_image(image_data, mask_data, patch_size=9):
        """
        Inpaint image using exemplar-based method
        
        Args:
            image_data: Base64 encoded image to inpaint
            mask_data: Base64 encoded binary mask where 1 indicates areas to inpaint
            patch_size: Size of patches to use for inpainting (default: 9)
            
        Returns:
            Base64 encoded inpainted image
        """
        try:
            # Decode base64 images
            image = ImageUtils.decode_base64_image(image_data)
            mask = ImageUtils.decode_base64_image(mask_data)
            
            # DEBUGGING
            # plt.imshow(mask)
            # plt.show()
            # return image
            
            # Create inpainter and process
            inpainter = Inpainter(image, mask, patch_size=patch_size)
            inpainter.initialize()
            inpainter.inpaint()
            
            # Return result
            return inpainter.image
            
        except Exception as e:
            print(f"Error in image inpainting: {str(e)}")
            raise