import cv2
from ..pipeline import FrameStage
import logging

logger = logging.getLogger(__name__)


class OverlayFPS(FrameStage):
    def __init__(self):
        self.last_t = None
        self.count = 0
        self.fps = 0.0


    def __call__(self, frame):
        import time
        t = time.time()
        self.count += 1
        if self.last_t is None:
            self.last_t = t
        elif t - self.last_t >= 0.5:
            self.fps = self.count / (t - self.last_t)
            self.last_t, self.count = t, 0
        
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {self.fps:.1f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return frame