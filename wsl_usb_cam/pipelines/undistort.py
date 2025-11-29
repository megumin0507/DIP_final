import cv2
import numpy as np
import logging
from ..pipeline import FrameStage


logger = logging.getLogger(__name__)


class Undistort(FrameStage):
    def __init__(self, calib_file: str):
        self.data = np.load(calib_file)
        self.K = self.data["camera_matrix"]
        self.dist = self.data["dist_coeffs"]


    def __call__(self, frame):
        h, w = frame.shape[:2]
        newK, roi = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), alpha=0)
        undistorted = cv2.undistort(frame, self.K, self.dist, None, newK)
        return undistorted