#!/usr/bin/env python3
"""MJPEG video server - stream USB camera over HTTP."""

import cv2
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

HOST = "0.0.0.0"
PORT = 8090
CAMERA = 0
FPS = 15
QUALITY = 70  # JPEG quality (1-100)

frame_lock = threading.Lock()
current_frame = None

def camera_loop():
    global current_frame
    cap = cv2.VideoCapture(CAMERA)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"[camera] Opened /dev/video{CAMERA}")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
        with frame_lock:
            current_frame = jpg.tobytes()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            # Simple HTML page with the stream
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b'<html><body style="margin:0;background:#000">'
                             b'<img src="/stream" style="width:100%;height:100vh;object-fit:contain">'
                             b'</body></html>')
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame = current_frame
                    if frame is None:
                        continue
                    self.wfile.write(b"--frame\r\n"
                                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/snap":
            with frame_lock:
                frame = current_frame
            if frame:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(frame)
            else:
                self.send_response(503)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress request logs

if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    server = HTTPServer((HOST, PORT), Handler)
    print(f"[server] Streaming at http://192.168.0.112:{PORT}")
    print(f"[server] /       - HTML viewer")
    print(f"[server] /stream - raw MJPEG stream")
    print(f"[server] /snap   - single JPEG snapshot")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
