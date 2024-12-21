from app.utils.image_utils import ImageUtils
from app.models.possion_editing import poisson_edit
import cv2

class PoissonService:
    @staticmethod
    def blend_images(source_data, mask_data, target_data, mode="Max"):
        """
        Blend source image into target image using Poisson blending
        
        Args:
            source_data: Base64 encoded source image
            mask_data: Base64 encoded mask image  
            target_data: Base64 encoded target image
            method: Blending method (currently ignored as only one method is supported)
            
        Returns:
            Base64 encoded blended image
        """
        # Decode base64 images
        source = ImageUtils.decode_base64_image(source_data)
        mask = ImageUtils.decode_base64_image(mask_data) 
        target = ImageUtils.decode_base64_image(target_data)
        
        # Convert mask to binary if needed
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask[mask > 0] = 255
        
        try:
            result = poisson_edit(source, target, mask, mode)
            return ImageUtils.encode_image(result)
            
        except Exception as e:
            print(f"Error in Poisson blending: {str(e)}")
            raise