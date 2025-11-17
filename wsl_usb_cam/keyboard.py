import sys, time, threading
from .app import AppCore
import logging

logger = logging.getLogger(__name__)


class KeyboardController:
    def __init__(self, app_core: AppCore):
        self.app_core = app_core
        self._stop = threading.Event()


    def start(self):
        logger.info("Starting keyboard controller...")
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        self._thread = t


    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Keyboard controller stopped")

    # --- internal

    def _loop(self):
        while not self._stop.is_set():
            line = sys.stdin.readline()
            if not line:
                time.sleep(0.1)
                continue
            c = line.strip().lower()
            if c == "v":
                self.app_core.request_capture()
            elif c == "q":
                logger.info("Stop requested")
                self.app_core.request_stop()
                break

