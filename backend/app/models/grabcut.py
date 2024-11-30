import numpy as np
import cv2 as cv
import datetime

class GrabCutManager:
    """
    Singleton class to manage the GrabCut algorithm for image segmentation.
    """

    _instance = None

    def __init__(self, img=None, rect=None):
        """
        Initialize the GrabCutManager with an image and a rectangle.
        
        :param img: The input image.
        :param rect: The rectangle for the initial segmentation.
        """
        if img is not None and rect is not None:
            self.init_grabcut(img, rect)
            GrabCutManager._instance = self

    @classmethod
    def get_instance(cls, img=None, rect=None):
        """
        Get the singleton instance of the GrabCutManager.
        
        :param img: The input image.
        :param rect: The rectangle for the initial segmentation.
        :return: The singleton instance of GrabCutManager.
        """
        if cls._instance is None:
            cls._instance = cls(img, rect)
        return cls._instance

    @classmethod
    def reset(cls):
        """
        Reset the singleton instance of the GrabCutManager.
        """
        cls._instance = None

    def init_grabcut(self, img, rect):
        """
        Initialize the GrabCut algorithm with an image and a rectangle.
        
        :param img: The input image.
        :param rect: The rectangle for the initial segmentation.
        """
        self.mask = np.zeros(img.shape[:2], np.uint8)
        self.bgdModel = np.zeros((1, 65), np.float64)
        self.fgdModel = np.zeros((1, 65), np.float64)
        self.rect = rect
        self.img = img
        self.first = True

    def apply_grabcut(self, mask=None, iter=1):
        """
        Apply the GrabCut algorithm to segment the image.
        
        :param mask: The mask to refine the segmentation.
        :param iter: The number of iterations for the algorithm.
        :return: The mask and the path of the largest contour.
        """
        if self.first:
            try:
                # Initialize GrabCut with rectangle
                cv.grabCut(self.img, self.mask, self.rect, self.bgdModel, self.fgdModel, iter, cv.GC_INIT_WITH_RECT)
                self.first = False
            except Exception as e:
                print("Error", str(e))
        else:
            try:
                # Convert mask to uint8
                mask = np.array(mask, dtype=np.uint8)
                print(mask.shape)
                self.mask = mask
                tempMask = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')

                # Initialize GrabCut with mask
                cv.grabCut(self.img, mask, self.rect, self.bgdModel, self.fgdModel, iter, cv.GC_INIT_WITH_MASK)
            except Exception as e:
                print("Error", str(e))

        try:
            # Create binary mask
            mask2 = np.where((self.mask == 1) + (self.mask == 3), 255, 0).astype('uint8')
            # Find contours in the binary mask
            contours, _ = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            path = []
            if len(contours) > 0:
                # Get largest contour
                largest_contour = max(contours, key=cv.contourArea)
                path = largest_contour.reshape(-1, 2).tolist()

            return self.mask.tolist(), path

        except Exception as e:
            print("Error processing contours:", str(e))
            return self.mask, []