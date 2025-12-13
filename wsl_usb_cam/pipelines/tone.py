import cv2
import numpy as np
from ..pipeline import FrameStage
import logging

logger = logging.getLogger(__name__)


class ToneAdjust(FrameStage):
    def __init__(self, brightness: float = 0.0, contrast: float = 1.0, saturation: float = 1.0, skin_tone_scale: float = 1.2):
        self.brightness = np.clip(float(brightness), -1.0, 1.0)
        self.contrast = np.clip(float(contrast), 0.5, 2.0)
        self.saturation = np.clip(float(saturation), 0.0, 2.0)
        self.skin_tone_scale = np.clip(float(skin_tone_scale), 0.0, 2.0)
        logger.info(f"ToneAdjust set brightness={self.brightness}, contrast={self.contrast}, saturation={self.saturation}" +
                    f", skin_tone_scale={self.skin_tone_scale}")


    def set_params(self, brightness=None, contrast=None, saturation=None, skin_tone_scale=None):
        if brightness is not None:
            self.brightness = np.clip(float(brightness), -1.0, 1.0)
        if contrast is not None:
            self.contrast = np.clip(float(contrast), 0.5, 2.0)
        if saturation is not None:
            self.saturation = np.clip(float(saturation), 0.0, 2.0)
        if skin_tone_scale is not None:
            self.skin_tone_scale = np.clip(float(skin_tone_scale), 0.0, 2.0)
        logger.debug(f"ToneAdjust updated: brightness={self.brightness}, contrast={self.contrast}," +
                     f" saturation={self.saturation}, skin_tone_scale={self.skin_tone_scale}")


    def __call__(self, frame):
        if frame is None:
            return frame

        tone_frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness * 255)

        hsv_frame = cv2.cvtColor(tone_frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_frame[..., 1] *= self.saturation
        hsv_frame[..., 1] = np.clip(hsv_frame[..., 1], 0, 255)
        tone_frame = cv2.cvtColor(hsv_frame.astype(np.uint8), cv2.COLOR_HSV2BGR)
        if self.skin_tone_scale != 1.0:
            tone_frame = SkinToneAdjust(self.skin_tone_scale)(tone_frame)
        return tone_frame

class LCE(FrameStage):
    """
    Local Contrast Enhancement (LCE) by large-area bilateral filtering.
    ref: https://www.dl-c.com/Temp/downloads/Whitepapers/Local%20Contrast%20Enhancement.pdf
            Local Contrast Enhancement
            Jonathan Sachs
            17-May-2016
    """
    def __init__(self, diameter: int = 41, sigma_color: float = 20.0, sigma_space: float = 7.0, amount: float = 0.5):
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.amount = amount
        logger.info(
            f"LCE initialized with diameter={diameter}, sigma_color={sigma_color}, sigma_space={sigma_space}, amount={amount}")
        
    def __call__(self, frame):
        if frame is None:
            return frame

        blur = cv2.bilateralFilter(frame, d=self.diameter, sigmaColor=self.sigma_color,
                                   sigmaSpace=self.sigma_space, borderType=cv2.BORDER_DEFAULT)
        lce_frame = cv2.addWeighted(frame, 1.0 + self.amount, blur, -self.amount, 0)

        return lce_frame
    
class SkinToneAdjust(FrameStage):
    def __init__(self, scale: float = 1.2):
        self.scale = scale

    def __call__(self, frame):
        if frame is None:
            return frame
        return self.adjust_skin_saturation_lab(frame, self.scale)

    def skin_mask_ycrcb(self, img_bgr):
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)

        mask = np.logical_and.reduce((
            Cr > 135, Cr < 180,
            Cb > 85,  Cb < 135
        ))
        return mask.astype(np.uint8)

    def adjust_skin_saturation_lab(self, img_bgr, scale=1.2):
        mask = self.skin_mask_ycrcb(img_bgr)
        # soft mask
        mask = cv2.GaussianBlur(mask.astype(np.float32), (15,15), 0)

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, a, b = cv2.split(lab)

        # Center at 128
        a -= 128
        b -= 128

        a = cv2.addWeighted(a * mask, scale, a * (1 - mask), 1.0, 0)
        b = cv2.addWeighted(b * mask, scale, b * (1 - mask), 1.0, 0)

        a += 128
        b += 128

        lab = cv2.merge([L, a, b])
        lab = np.clip(lab, 0, 255).astype(np.uint8)

        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)