import torch
from app.models.DNN import process_image
from app.utils.image_utils import ImageUtils
import cv2
from pathlib import Path

class DNNService:
    _instance = None
    _model = None
    _device = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if DNNService._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            DNNService._instance = self
            self._initialize_model()

    def _initialize_model(self):
        try:
            self._device = torch.device("cpu")
            model_path = Path(__file__).parent.parent / "assets" / "big-lama.pt"
            self._model = torch.jit.load(str(model_path), map_location="cpu")
            self._model = self._model.to(self._device)
            self._model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    def inpaint_with_dnn(self, image_data, mask_data):
        """
        Inpaint image using the DNN model.
        
        Parameters:
        image_data (str): Base64 encoded image data
        mask_data (str): Base64 encoded mask data
        
        Returns:
        np.ndarray: Inpainted image
        """
        # Decode base64 image data
        image = ImageUtils.decode_base64_image(image_data)
        
        # Decode and process the mask
        mask = ImageUtils.decode_base64_image(mask_data)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        print(f"mask values statistics: {mask.min()}, {mask.max()}")
        mask[mask > 0] = 255
        
        # Process the image using DNN model
        result = process_image(image, mask, self._model, self._device)
        
        return result 