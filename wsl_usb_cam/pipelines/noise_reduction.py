# wsl_usb_cam/pipelines/noise_reduction.py

import numpy as np
import cv2
from ..pipeline import FrameStage
import logging

logger = logging.getLogger(__name__)


def _bilateral_filter_single_channel(img: np.ndarray,
                                     ksize: int = 7,
                                     sigma_s: float = 3.0,
                                     sigma_r: float = 0.08) -> np.ndarray:

    img = img.astype(np.float32)
    H, W = img.shape
    r = ksize // 2

    ax = np.arange(-r, r + 1)
    xx, yy = np.meshgrid(ax, ax)
    w_s = np.exp(-(xx**2 + yy**2) / (2 * sigma_s**2)).astype(np.float32)

    out = np.zeros_like(img, dtype=np.float32)

    # reflect padding 避免邊界 artifacts
    padded = np.pad(img, ((r, r), (r, r)), mode='reflect')

    for y in range(H):
        for x in range(W):
            patch = padded[y:y + ksize, x:x + ksize]
            center_val = img[y, x]

            diff = patch - center_val
            w_r = np.exp(-(diff**2) / (2 * sigma_r**2)).astype(np.float32)

            w = w_s * w_r
            out[y, x] = np.sum(w * patch) / (np.sum(w) + 1e-12)

    return out


class BilateralSmoothing(FrameStage):


    def __init__(self,
                 ksize: int = 7,
                 sigma_s: float = 3.0,
                 sigma_r: float = 0.08,
                 apply_to: str = "all"):

        self.ksize = int(ksize)
        self.sigma_s = float(sigma_s)
        self.sigma_r = float(sigma_r)
        self.apply_to = apply_to

        if self.ksize % 2 == 0 or self.ksize <= 1:
            raise ValueError("ksize 必須是大於 1 的奇數")

    def __call__(self, frame):

        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return frame

        if self.apply_to == "all":
            return self._apply_to_all_channels(frame)
        else:
            logger.warning(f"BilateralSmoothing: 未知 apply_to={self.apply_to}，不做處理")
            return frame

    # ---------------- internal helpers ----------------

    def _apply_to_all_channels(self, frame_bgr_uint8: np.ndarray) -> np.ndarray:

        img01 = frame_bgr_uint8.astype(np.float32) / 255.0  # [0,1] float
        b, g, r = cv2.split(img01)

        b_f = _bilateral_filter_single_channel(b, ksize=self.ksize, sigma_s=self.sigma_s, sigma_r=self.sigma_r)
        g_f = _bilateral_filter_single_channel(g, ksize=self.ksize, sigma_s=self.sigma_s, sigma_r=self.sigma_r)
        r_f = _bilateral_filter_single_channel(r, ksize=self.ksize, sigma_s=self.sigma_s, sigma_r=self.sigma_r)

        out01 = cv2.merge([b_f, g_f, r_f])
        out = np.clip(out01 * 255.0, 0, 255).astype(np.uint8)
        return out
