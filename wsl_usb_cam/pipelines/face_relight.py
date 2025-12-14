import numpy as np
import cv2
import os
from ..pipeline import FrameStage
import logging

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False
    logger.error("mediapipe ，FaceRelight 將不會啟用，請先執行：pip install mediapipe")


class FaceRelight(FrameStage):
    """
    Ref: bg_blur_seg.py
    先將影像 downscale 後再跑 segmentation、可設定每 N 張 frame 才重算一次 mask。
    """

    def __init__(self,
                 downscale: float = 0.1,
                 update_every: int = 2,
                 alpha: float = 1.5,
                 sigma: float = 0.45):
        """
        Docstring for __init__
        
        :param alpha: Brightness factor
        :type alpha: float
        :param sigma: Sigma for mask size
        :type sigma: float
        """

        self.enabled = _HAS_MEDIAPIPE
        if not self.enabled:
            logger.warning("FaceRelight 已建立，但 mediapipe 不存在，將直接回傳原始畫面。")
            return

        self.downscale = float(downscale)
        self.update_every = max(1, int(update_every))
        self.alpha = float(alpha)
        self.sigma = float(sigma)

        self.frame_count = 0
        self._last_mask = None  # float32, [H,W], 0~1


        model_path = os.path.join(os.path.dirname(__file__), 'blaze_face_short_range.tflite')
        with open(model_path, "rb") as f:
            model_buffer = f.read()
        base_options = python.BaseOptions(model_asset_buffer=model_buffer)

        options = vision.FaceDetectorOptions(base_options=base_options)
        self.face_detector = vision.FaceDetector.create_from_options(options)

        logger.info(
            f"FaceRelight 初始化: "
            f"downscale={self.downscale}, update_every={self.update_every}, "
            f"alpha={alpha}, sigma={sigma}"
        )

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
                logger.warning(f"FaceRelight: 計算 face segmentation 失敗，使用上一張 mask。err={e}")
                if self._last_mask is None:
                    return frame  # 沒有 mask 可以用，只好直接回傳原圖
        else:
            mask = self._last_mask

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)
        L *= (1.0 + self.alpha * mask)
        np.clip(L, 0, 255, out=L)

        lab[:, :, 0] = L.astype(np.uint8)   
        frame = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return frame


    # ---------------- internal helpers ----------------

    def _compute_mask(self, frame_bgr_uint8: np.ndarray) -> np.ndarray:
        h, w = frame_bgr_uint8.shape[:2]

        # downscale for speed
        if 0 < self.downscale < 1.0:
            small = cv2.resize(frame_bgr_uint8, None, fx=self.downscale, fy=self.downscale,
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = frame_bgr_uint8

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

        bbox = self.face_detector.detect(mp_image).detections[0].bounding_box

        # Face center (pixel)
        if 0 < self.downscale < 1.0:
            cx = int((bbox.origin_x + bbox.width / 2) / self.downscale)
            cy = int((bbox.origin_y + bbox.height / 2) / self.downscale)
            face_size = int((bbox.width + bbox.height) / 2 / self.downscale)
        else:
            cx = bbox.origin_x + bbox.width // 2
            cy = bbox.origin_y + bbox.height // 2
            face_size = (bbox.width + bbox.height) // 2
    
        sigma = self.sigma * face_size
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, center=(cx, cy), radius=(face_size // 4), color=(1.0,), thickness=-1)
        kernel_size = max(int(sigma * 6) | 1, 3)  # ensure odd and at least 3
        cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma, dst=mask)
        return mask