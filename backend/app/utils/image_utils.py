import base64
import cv2
import numpy as np

class ImageUtils:
    @staticmethod
    def decode_base64_image(base64_string: str) -> np.ndarray:
        """Decode base64 string to OpenCV image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64 to bytes
            img_data = base64.b64decode(base64_string)
            
            # Convert to numpy array
            nparr = np.frombuffer(img_data, np.uint8)
            
            # Decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Failed to decode image")
                
            return img
            
        except Exception as e:
            raise ValueError(f"Image decoding failed: {str(e)}")

    # expectes a numpy array, np.unit8, RGB format
    @staticmethod
    def encode_image(image: np.ndarray, format: str = 'png') -> str:
        """Encode OpenCV image (numpy array) to base64 string"""
        try:
            # Encode image to bytes
            _, buffer = cv2.imencode(f'.{format}', image)
            img_data = buffer.tobytes()

            # Encode bytes to base64 string
            base64_string = base64.b64encode(img_data).decode('utf-8')

            # Optionally, prepend the data URL prefix
            data_url = f'data:image/{format};base64,' + base64_string

            return data_url

        except Exception as e:
            raise ValueError(f"Image encoding failed: {str(e)}")