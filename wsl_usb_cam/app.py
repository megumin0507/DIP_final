import threading
from .camera import CameraService
from .pipeline import Pipeline
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AppCore:
    def __init__(self, camera_cfg, pipeline: Pipeline):
        self.camera = CameraService(**camera_cfg)
        self.pipeline = pipeline

        self.latest_frame = None
        self.latest_lock = threading.Lock()

        self.want_capture = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self.save_dir = Path("./captures/")
        self.jpeg_quality = 90

    
    def start(self):
        logger.info("Starting AppCore...")
        self.camera.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()


    def stop(self):
        self._stop.set()
        self.camera.stop()
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("AppCore stopped")


    def get_latest(self):
        with self.latest_lock:
            return None if self.latest_frame is None else self.latest_frame.copy()
        

    def request_capture(self):
        self.want_capture.set()

    
    def request_stop(self):
        self._stop.set()


    # --- internal:

    def _loop(self):
        while not self._stop.is_set():
            frame = self.camera.frame_queue.get()
            if frame is None:
                continue

            frame = self.pipeline.process(frame)

            with self.latest_lock:
                self.latest_frame = frame.copy()

            if self.want_capture.is_set():
                self._capture(frame)
                

    def _capture(self, frame):
        from datetime import datetime
        import cv2

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        out = self.save_dir / f"capture_{ts}.jpg"

        self.save_dir.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if success:
            logger.info(f"Image successfully saved: {out}")
        else:
            logger.error(f"Failed to save image: {out}")
        self.want_capture.clear()
