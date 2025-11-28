from wsl_usb_cam.pipeline import Pipeline
from wsl_usb_cam.pipelines.demo import OverlayFPS
from wsl_usb_cam.pipelines.awb import AutoWhiteBalance
from wsl_usb_cam.pipelines.noise_reduction import BilateralSmoothing
from wsl_usb_cam.app import AppCore
from wsl_usb_cam.web import create_app
from wsl_usb_cam.keyboard import KeyboardController
import threading, time
import logging


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    camera_cfg = {"device_index": 3, "size": (640, 480), "fps": 30}

    # later add more stages here
    pipeline = Pipeline([
        AutoWhiteBalance(p=1.0, ksize=3, update_every=5),
        BilateralSmoothing(ksize=7, sigma_s=3.0, sigma_r=0.08, apply_to="all"),
        OverlayFPS(),
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