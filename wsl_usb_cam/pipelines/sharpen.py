import cv2
import numpy as np
from ..pipeline import FrameStage
import logging

logger = logging.getLogger(__name__)

class Sharpen(FrameStage):
    """
    Unsharp Masking
    """
    def __init__(self, amount: float = 1.0):
        self.amount = amount
        logger.info(f"Sharpen initialized with amount={amount}")

    def __call__(self, frame):
        if frame is None:
            return frame

        blurred = cv2.bilateralFilter(frame, d=3, sigmaColor=50, sigmaSpace=7)
        sharpened = cv2.addWeighted(frame, 1 + self.amount, blurred, -self.amount, 0)
        return sharpened