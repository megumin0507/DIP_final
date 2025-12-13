from flask import Flask, Response, render_template, render_template_string, request
from .app import AppCore
from .pipelines.tone import ToneAdjust
import cv2, time
import logging

logger = logging.getLogger(__name__)

def create_app(app_core: AppCore):
    logger.info("Starting creating app...")
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")
    
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

    @app.route("/pipeline_config", methods=["POST"])
    def pipeline_config():
        data = request.get_json(force=True) or {}
        stages = data.get("stages", {})
        app_core.set_stage_config(stages)
        return "", 204

    @app.route("/tone_update", methods=["POST"])
    def tone_update():
        data = request.get_json(force=True) or {}
        if data is not None:
            app_core.set_tone_params(data)
        return "", 204
    
    return app