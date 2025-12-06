import cv2
import numpy as np
from ..pipeline import FrameStage
import logging

logger = logging.getLogger(__name__)


class VideoStabilizer(FrameStage):
    """
    Simple online stabilizer:
    - detects corners in previous frame
    - tracks them to current frame (LK optical flow)
    - estimates affine transform (translation + rotation)
    - smooths that motion with EMA, then warps current frame
    """

    def __init__(
            self,
            max_corners: int = 200,
            quality_level: float = 0.01,
            min_distance: int = 30,
            ema_alpha: float = 0.9
    ):
        self.prev_gray = None

        self.ema_alpha = float(ema_alpha)
        self.smooth_dx = 0.0
        self.smooth_dy = 0.0
        self.smooth_da = 0.0  # rotation

        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=3
        )

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    
    def _estimate_transform(self, prev_gray, gray):
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)
        if p0 is None or len(p0) == 0:
            return 0.0, 0.0, 0.0
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **self.lk_params)
        if p1 is None or st is None:
            return 0.0, 0.0, 0.0
        
        st = st.reshape(-1)
        p0 = p0[st == 1]
        p1 = p1[st == 1]

        if len(p0) < 8:
            return 0.0, 0.0, 0.0
        
        M, inliers = cv2.estimateAffinePartial2D(p0, p1)
        if M is None:
            return 0.0, 0.0, 0.0
        
        dx = float(M[0, 2])
        dy = float(M[1, 2])
        da = float(np.arctan2(M[1, 0], M[0, 0]))

        return dx, dy, da
    

    def __call__(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return frame
        
        dx, dy, da = self._estimate_transform(self.prev_gray, gray)

        a = self.ema_alpha
        self.smooth_dx = a * self.smooth_dx + (1.0 - a) * dx
        self.smooth_dy = a * self.smooth_dy + (1.0 - a) * dy
        self.smooth_da = a * self.smooth_da + (1.0 - a) * da

        diff_dx =  self.smooth_dx - dx
        diff_dy =  self.smooth_dy - dy
        diff_da =  self.smooth_da - da

        cos_a = np.cos(diff_da)
        sin_a = np.sin(diff_da)

        M = np.array(
            [
                [cos_a, -sin_a, diff_dx],
                [sin_a, cos_a, diff_dy],
            ],
            dtype=np.float32
        )

        h, w = frame.shape[:2]
        stabilized = cv2.warpAffine(
            frame,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        self.prev_gray = gray
        return stabilized
        