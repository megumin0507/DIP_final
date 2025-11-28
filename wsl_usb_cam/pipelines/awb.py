# wsl_usb_cam/pipelines/awb.py
import cv2
import numpy as np
from ..pipeline import FrameStage
import logging

logger = logging.getLogger(__name__)


class AutoWhiteBalance(FrameStage):

    def __init__(self, p: float = 1.0, ksize: int = 3, update_every: int = 1):

        self.p = float(p)
        self.ksize = int(ksize)
        self.update_every = max(1, int(update_every))

        self._frame_count = 0
        # 初始化 gains，先給 1，代表不改變顏色
        self._gains = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def __call__(self, frame):

        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return frame

        self._frame_count += 1

        # 每 update_every 張 frame 才重新估計 illuminant
        if (self._frame_count % self.update_every) == 1:
            try:
                illum_est = self._grey_edge_estimate_illuminant(frame)
                gains = self._compute_gains_from_illuminant(illum_est)
                self._gains = gains
            except Exception as e:
                logger.warning(f"GrayEdgeAWB: failed to update illuminant, use previous gains. err={e}")

        # 套用目前的 gains
        awb_frame = self._apply_gains(frame, self._gains)
        return awb_frame

    def _grey_edge_estimate_illuminant(self, frame_bgr_uint8: np.ndarray) -> np.ndarray:

        # 轉成 float32 並 normalize 到 [0,1]
        img = frame_bgr_uint8.astype(np.float32) / 255.0
        eps = 1e-6
        e = []

        for c in range(3):
            I = img[..., c]
            gx = cv2.Sobel(I, cv2.CV_32F, 1, 0, ksize=self.ksize)
            gy = cv2.Sobel(I, cv2.CV_32F, 0, 1, ksize=self.ksize)
            grad_mag = np.hypot(gx, gy)  # sqrt(gx^2 + gy^2)

            val = np.power(np.mean(np.power(grad_mag + eps, self.p)), 1.0 / self.p)
            e.append(float(val))

        illum = np.array(e, dtype=np.float32)
        return illum

    def _compute_gains_from_illuminant(self, illum_est: np.ndarray) -> np.ndarray:

        means = illum_est.astype(np.float32)
        m = float(np.mean(means) + 1e-12)
        gains = m / (means + 1e-12)  # 讓三個 channel 的 "illum value" 拉到一樣
        return gains

    def _apply_gains(self, frame_bgr_uint8: np.ndarray, gains: np.ndarray) -> np.ndarray:

        img = frame_bgr_uint8.astype(np.float32)
        img[..., 0] *= float(gains[0])  # B
        img[..., 1] *= float(gains[1])  # G
        img[..., 2] *= float(gains[2])  # R

        img = np.clip(img, 0.0, 255.0).astype(np.uint8)
        return img