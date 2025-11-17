from flask import Flask, Response, render_template_string
from .app import AppCore
import cv2, time
import logging

logger = logging.getLogger(__name__)


HTML = """
<!doctype html>
<title>WSL USB Cam Preview</title>
<style>
body { margin:0; background:#111; color:#eee; font-family:system-ui, sans-serif; }
.bar { position:fixed; top:0; left:0; right:0; padding:10px 12px; background:rgba(0,0,0,.6); }
img { display:block; margin:0 auto; width: min(100vw, 1280px); height:auto; }
.wrap { padding-top:50px; }
kbd { background:#333; padding:2px 6px; border-radius:4px; }
</style>
<div class="bar">預覽中：在 <kbd>終端</kbd> 輸入 <kbd>v</kbd> 拍照、<kbd>q</kbd> 結束。</div>
<div class="wrap">
  <img src="/stream" />
</div>
"""

def create_app(app_core: AppCore):
    logger.info("Starting creating app...")
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(HTML)
    
    @app.route("/stream")
    def stream():
        def gen():
            while not app_core._stop.is_set():
                frame = app_core.get_latest()
                if frame is None:
                    time.sleep(0.03)
                    continue
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    continue
                jpg = buf.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    jpg +
                    b"\r\n"
                )
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app