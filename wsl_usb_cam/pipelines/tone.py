import cv2
import numpy as np
from ..pipeline import FrameStage
import logging

logger = logging.getLogger(__name__)


class ToneAdjust(FrameStage):
    def __init__(self, brightness: float = 0.0, contrast: float = 1.0, saturation: float = 1.0):
        self.brightness = np.clip(float(brightness), -1.0, 1.0)
        self.contrast = np.clip(float(contrast), 0.5, 2.0)
        self.saturation = np.clip(float(saturation), 0.0, 2.0)
        logger.info(f"ToneAdjust set brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}")


    def set_params(self, brightness=None, contrast=None, saturation=None):
        if brightness is not None:
            self.brightness = np.clip(float(brightness), -1.0, 1.0)
        if contrast is not None:
            self.contrast = np.clip(float(contrast), 0.5, 2.0)
        if saturation is not None:
            self.saturation = np.clip(float(saturation), 0.0, 2.0)
        logger.debug(f"ToneAdjust updated: brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}")


    def __call__(self, frame):
        if frame is None:
            return frame

        tone_frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness * 255)

        hsv_frame = cv2.cvtColor(tone_frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_frame[..., 1] *= self.saturation
        hsv_frame[..., 1] = np.clip(hsv_frame[..., 1], 0, 255)
        tone_frame = cv2.cvtColor(hsv_frame.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return tone_frame