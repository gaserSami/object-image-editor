from app.utils.image_utils import ImageUtils
from app.models.grabcut import GrabCutManager

class ImageService:
    @staticmethod
    def start_selection(image_data, rect, iter):
        """
        Process image selection using GrabCut algorithm.

        Args:
            image_data (str): Base64 encoded image data.
            rect (dict): Dictionary containing the rectangle coordinates and dimensions.
            iter (int): Number of iterations for the GrabCut algorithm.

        Returns:
            tuple: A tuple containing the mask and the path to the processed image.
        """
        # Decode base64 image
        img = ImageUtils.decode_base64_image(image_data)
        print(type(img))
        
        # Reset the GrabCutManager
        GrabCutManager.reset()
        
        # Convert rect dictionary to tuple
        rect_tuple = (
            int(rect["x"]), 
            int(rect["y"]), 
            int(rect["width"]), 
            int(rect["height"])
        )
        
        # Get an instance of GrabCutManager with the image and rectangle
        grabcut = GrabCutManager.get_instance(img, rect_tuple)
        
        # Apply grabcut algorithm
        mask, path = grabcut.apply_grabcut(None, iter)
        
        return mask, path
    
    @staticmethod
    def refine_selection(mask, iter):
        """
        Refine the image selection using the GrabCut algorithm.

        Args:
            mask (numpy.ndarray): The mask to refine.
            iter (int): Number of iterations for the GrabCut algorithm.

        Returns:
            tuple: A tuple containing the refined mask and the path to the processed image.
        """
        # Get an instance of GrabCutManager
        grabcut = GrabCutManager.get_instance()
        
        # Apply grabcut algorithm
        mask, path = grabcut.apply_grabcut(mask, iter)
        
        return mask, path