# wsl_usb_cam/pipelines/bg_blur_seg.py

import numpy as np
import cv2
from ..pipeline import FrameStage
import logging

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False
    logger.error("mediapipe 未安裝，BackgroundBlurSegmentation 將不會啟用，請先執行：pip install mediapipe")


class BackgroundBlurSegmentation(FrameStage):

    # 先將影像 downscale 後再跑 segmentation、可設定每 N 張 frame 才重算一次 mask。


    def __init__(self,
                 model_selection: int = 1,
                 downscale: float = 0.1,
                 update_every: int = 2,
                 foreground_threshold: float = 0.5,
                 mask_smooth_ksize: int = 21,
                 blur_ksize: int = 31,
                 blur_sigma: float = 0.0):
        
        self.enabled = _HAS_MEDIAPIPE
        if not self.enabled:
            logger.warning("BackgroundBlurSegmentation 已建立，但 mediapipe 不存在，將直接回傳原始畫面。")
            return

        self.downscale = float(downscale)
        self.update_every = max(1, int(update_every))
        self.foreground_threshold = float(foreground_threshold)

        # 調整 kernel size 為合法奇數
        self.mask_smooth_ksize = self._ensure_odd(max(3, int(mask_smooth_ksize)))
        self.blur_ksize = self._ensure_odd(max(3, int(blur_ksize)))
        self.blur_sigma = float(blur_sigma)

        self.frame_count = 0
        self._last_mask = None  # float32, [H,W], 0~1

        mp_selfie = mp.solutions.selfie_segmentation
        self.segmentor = mp_selfie.SelfieSegmentation(model_selection=model_selection)

        logger.info(
            f"BackgroundBlurSegmentation 初始化: "
            f"downscale={self.downscale}, update_every={self.update_every}, "
            f"mask_smooth_ksize={self.mask_smooth_ksize}, blur_ksize={self.blur_ksize}"
        )

    @staticmethod
    def _ensure_odd(k: int) -> int:
        return k if (k % 2 == 1) else k + 1

    def __call__(self, frame):

        if not self.enabled:
            return frame

        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return frame

        self.frame_count += 1
        H, W = frame.shape[:2]

        # 決定是否要更新 segmentation mask
        need_update = (self._last_mask is None) or (self.frame_count % self.update_every == 1)

        if need_update:
            try:
                mask = self._compute_mask(frame)
                self._last_mask = mask
            except Exception as e:
                logger.warning(f"BackgroundBlurSegmentation: 計算 segmentation mask 失敗，使用上一張 mask。err={e}")
                if self._last_mask is None:
                    return frame  # 沒有 mask 可以用，只好直接回傳原圖
        else:
            mask = self._last_mask

        # 背景高斯模糊
        blurred = cv2.GaussianBlur(frame, (self.blur_ksize, self.blur_ksize), self.blur_sigma)

        # ===== 新增：前景（人像）專用 bilateral + sharpen =====
        # 先對整張圖算（效率較好）
        bilateral = cv2.bilateralFilter(frame, d=3, sigmaColor=50, sigmaSpace=7)

        # Unsharp masking（amount 固定寫死或改成參數）
        amount = 1.0
        sharpened_fg = cv2.addWeighted(
            frame, 1 + amount,
            bilateral, -amount,
            0
        )

        # 將 mask 擴展成 3 channel，做 alpha blending
        # mask: 前景=1, 背景=0
        mask3 = np.repeat(mask[:, :, None], 3, axis=2).astype(np.float32)

        # fg = frame.astype(np.float32)
        fg = sharpened_fg.astype(np.float32)
        bg = blurred.astype(np.float32)

        out = fg * mask3 + bg * (1.0 - mask3)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out

    # ---------------- internal helpers ----------------

    def _compute_mask(self, frame_bgr_uint8: np.ndarray) -> np.ndarray:
   
        H, W = frame_bgr_uint8.shape[:2]

        # downscale for speed
        if 0 < self.downscale < 1.0:
            small = cv2.resize(frame_bgr_uint8, None, fx=self.downscale, fy=self.downscale,
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = frame_bgr_uint8

        # BGR -> RGB
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Mediapipe inference
        res = self.segmentor.process(rgb)
        seg = res.segmentation_mask  # float32, [h,w], 值域大致在 [0,1]

        # resize 回原尺寸
        seg_full = cv2.resize(seg, (W, H), interpolation=cv2.INTER_LINEAR)

        # 產生 mask：大於 threshold 的視為人像
        mask = (seg_full > self.foreground_threshold).astype(np.float32)

        if self.mask_smooth_ksize >= 3:
            mask = cv2.GaussianBlur(mask, (self.mask_smooth_ksize, self.mask_smooth_ksize), 0)

        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        return mask
