from wsl_usb_cam.pipeline import Pipeline
from wsl_usb_cam.pipelines.demo import OverlayFPS
from wsl_usb_cam.pipelines.awb import AutoWhiteBalance
from wsl_usb_cam.pipelines.noise_reduction import BilateralSmoothing
from wsl_usb_cam.pipelines.tone import ToneAdjust, LCE
from wsl_usb_cam.pipelines.undistort import Undistort
from wsl_usb_cam.pipelines.bg_blur_seg import BackgroundBlurSegmentation
from wsl_usb_cam.pipelines.stabilize import VideoStabilizer
from wsl_usb_cam.app import AppCore
from wsl_usb_cam.web import create_app
from wsl_usb_cam.keyboard import KeyboardController
import threading, time
import logging


DEVICE_INDEX = 2 # Change this to your own device index


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    camera_cfg = {"device_index": DEVICE_INDEX, "size": (640, 480), "fps": 30}

    # later add more stages here
    pipeline = Pipeline([
    ("bilateral",   BilateralSmoothing(diameter=7, sigma_color=25, sigma_space=5)),
    ("awb",         AutoWhiteBalance(p=1.0, ksize=3, update_every=5)),
    ("undistort",   Undistort(calib_file="calibration_result.npz")),
    ("stabilize",   VideoStabilizer()),
    ("lce",         LCE()),
    ("tone",        ToneAdjust()),
    ("background_blur",        BackgroundBlurSegmentation(
            downscale=0.1,          # 可調速度
            update_every=2,         # 每 x 個 frame 更新 mask
            foreground_threshold=0.5,
            mask_smooth_ksize=21,
            blur_ksize=31,          # Blur 程度
            blur_sigma=0.0
        )),
    ("overlay_fps", OverlayFPS()),
    ])

    core = AppCore(camera_cfg=camera_cfg, pipeline=pipeline)
    kb = KeyboardController(core)

    core.start()
    kb.start()

    flask_app = create_app(core)
    t_flask = threading.Thread(
        target=lambda: flask_app.run(host="127.0.0.1", port=5000, threaded=True),
        daemon=True
    )
    t_flask.start()

    try:
        while not core._stop.is_set():
            time.sleep(0.1)
    finally:
        logger.info("Shutting down...")
        kb.stop()
        core.stop()


if __name__ == "__main__":
    main()