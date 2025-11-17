#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSL USB Camera with Browser Preview (MJPEG) & Terminal-triggered Capture
- 在 http://127.0.0.1:5000/ 預覽，終端輸入 'v' 拍照、'q' 結束
"""

import os
import sys
import cv2
import time
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, Response, render_template_string

# ------------------ 設定 ------------------
DEVICE_INDEX = 2
REQ_W, REQ_H = 640, 480
SAVE_DIR = Path("./captures")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
JPEG_QUALITY = 90

latest_frame_lock = threading.Lock()
latest_frame = None
want_quit = threading.Event()
want_capture = threading.Event()

def _open_cap_with_fourcc(dev, w, h, fps, fourcc_str):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
 
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    if w > 0: cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    if h > 0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if fps:   cap.set(cv2.CAP_PROP_FPS, float(fps))

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    r_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    r_fourcc_str = "".join([chr((r_fourcc >> 8*i) & 0xFF) for i in range(4)])
    afps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Opened /dev/video{dev} with FOURCC={r_fourcc_str} @ {aw}x{ah} {afps:.1f}fps", flush=True)
    return cap

def camera_thread():
    global latest_frame
    dev = DEVICE_INDEX
    # 1) MJPG
    cap = _open_cap_with_fourcc(dev, REQ_W, REQ_H, 30, "YUYV")
    if cap is None:
        print("[WARN] YUYV 失敗，改試 MPEG...", flush=True)
        cap = _open_cap_with_fourcc(dev, REQ_W, REQ_H, 30, "MPEG")
    if cap is None:
        print("[ERR] 無法以 MJPG/YUYV 開啟攝影機，請用 v4l2-ctl 檢查支援格式。", flush=True)
        want_quit.set()
        return

    print("[INFO] Camera started", flush=True)
    t0 = time.time(); cnt = 0; fps = 0.0
    while not want_quit.is_set():
        ok, frame = cap.read()
        if not ok or frame is None:
            # avoid select timeout
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        cnt += 1
        t1 = time.time()
        if t1 - t0 >= 0.5:
            fps = cnt / (t1 - t0)
            t0, cnt = t1, 0

        # 疊資訊
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(frame, "Open http://127.0.0.1:5000/ | Type 'v' in TERMINAL to capture, 'q' to quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-150, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,255,180), 1, cv2.LINE_AA)
        #cv2.drawMarker(frame, (w//2, h//2), (255,255,255), cv2.MARKER_CROSS, 18, 2)

        if want_capture.is_set():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            out = SAVE_DIR / f"capture_{ts}.jpg"
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            print(f"[CAPTURE] Saved: {out}", flush=True)
            want_capture.clear()

        with latest_frame_lock:
            latest_frame = frame.copy()

    cap.release()
    print("[INFO] Camera stopped", flush=True)

def keyboard_thread():
    raw_mode_ok = False
    if os.name == "posix" and sys.stdin.isatty():
        try:
            import termios, tty
            global _orig_attr
            _orig_attr = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            raw_mode_ok = True
        except Exception:
            raw_mode_ok = False

    try:
        if raw_mode_ok:
            import select
            while not want_quit.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    ch = sys.stdin.read(1)
                    if not ch: continue
                    c = ch.strip().lower()
                    if c == 'v': want_capture.set()
                    elif c == 'q':
                        want_quit.set(); break
        else:
            while not want_quit.is_set():
                line = sys.stdin.readline()
                if not line: time.sleep(0.1); continue
                c = line.strip().lower()
                if c == 'v': want_capture.set()
                elif c == 'q':
                    want_quit.set(); break
    finally:
        if raw_mode_ok:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _orig_attr)

# Flask 伺服器（MJPEG）
app = Flask(__name__)

INDEX_HTML = """
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

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

def mjpeg_generator():
    while not want_quit.is_set():
        with latest_frame_lock:
            frame = None if (latest_frame is None) else latest_frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        b = jpeg.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + b + b"\r\n")
        time.sleep(0.01)

@app.route("/stream")
def stream():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def main():

    t_cam = threading.Thread(target=camera_thread, daemon=True)
    t_key = threading.Thread(target=keyboard_thread, daemon=True)
    t_cam.start(); t_key.start()

    print("[INFO] 打開瀏覽器 http://127.0.0.1:5000/ 預覽", flush=True)
    try:
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
    finally:
        want_quit.set()
        t_cam.join(timeout=2)
        t_key.join(timeout=2)
        print("[INFO] 已結束。", flush=True)

if __name__ == "__main__":
    main()
