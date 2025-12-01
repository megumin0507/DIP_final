# wsl_usb_cam/pipelines/noise_reduction.py

import numpy as np
import cv2
from ..pipeline import FrameStage
import logging

logger = logging.getLogger(__name__)


class BilateralSmoothing(FrameStage):

    def __init__(self,
                 diameter: int = 7,
                 sigma_color: float = 20.0,
                 sigma_space: float = 5.0):

        self.diameter = int(diameter)
        self.sigma_color = float(sigma_color)
        self.sigma_space = float(sigma_space)

    def __call__(self, frame):

        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return frame

        # BGR uint8
        out = cv2.bilateralFilter(
            frame,
            d=self.diameter,
            sigmaColor=self.sigma_color,
            sigmaSpace=self.sigma_space
        )
        return out
