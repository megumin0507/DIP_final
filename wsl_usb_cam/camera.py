import cv2, time, threading
import logging
from queue import Queue

logger = logging.getLogger(__name__)

class CameraService:
    def __init__(self, device_index, size, fps, fourcc_list=('MJPG', "YUYV")):
        self.device_index = device_index
        self.req_w, self.req_h = size
        self.req_fps = fps
        self.fourcc_list = fourcc_list

        self.frame_queue = Queue(maxsize=5)
        self._stop = threading.Event()
        self._thread = None
        self.cap = None


    def start(self):
        logger.info("Starting camera service...")
        self.cap = self._open_camera()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()


    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        logger.info("Camera service stopped")


    # --- internal helpers:

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Exception:
                    pass
            self.frame_queue.put(frame)


    def _open_camera(self):
        cap = self._open_cap_with_fourcc(self.device_index, self.req_w, self.req_h, 30, "YUYV")
        if cap is None:
            logger.warning("Failed to open camera with YUYV, trying MJPG...")
            cap = self._open_cap_with_fourcc(self.device_index, self.req_w, self.req_h, 30, "MJPG")

        if cap is None:
            logger.error("Can't open camera with YUYV/MJPG, please check supported format using v4l2-ctl")
            self.stop()
            return None
        
        logger.info("Camera started")
        return cap
    

    def _open_cap_with_fourcc(self, dev, w, h, fps, fourcc_str):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            return None
        
        self._configure_capture_properties(cap, w, h, fps, fourcc_str)
        if not self._probe_capture_info(cap, dev):
            cap.release()
            return None
        return cap
    

    def _configure_capture_properties(self, cap, w, h, fps, fourcc_str):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        if w > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        if h > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, float(fps))


    def _probe_capture_info(self, cap, dev):
        ok, frame = cap.read()
        if not ok or frame is None:
            return False
        
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        r_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        r_fourcc_str = "".join([chr((r_fourcc >> 8 * i) & 0xFF) for i in range(4)])
        afps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Opened /dev/video{dev} with FOURCC={r_fourcc_str}")
        logger.info(f"@ {aw}x{ah} {afps:.1f}fps")
        return True
