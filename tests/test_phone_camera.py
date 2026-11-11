"""Phone camera over Wi-Fi, tested against a simulated DroidCam app (MJPEG stream at /video)."""
import http.server
import socket
import threading
import time

import cv2
import numpy as np
import pytest
from conftest import login

from invoice_parser import camera
from invoice_parser.camera import CameraError, PhoneStream


def _jpeg(shade, size=(120, 160)):
    img = np.full((*size, 3), shade, np.uint8)
    cv2.putText(img, "INV", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _scrambled_jpeg():
    """Like DroidCam's obscured feed: alternating dark and green scan lines."""
    img = np.zeros((120, 160, 3), np.uint8)
    img[1::2] = (0, 255, 0)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


class _FakeDroidCam(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/video") and self.server.busy and "/force" not in self.path:
            # like the real app: a styled page whose "Busy" text comes after a long <head>
            body = (b"<!doctype html>\n<html>\n<head>\n\t<meta charset=\"utf-8\">\n<style>"
                    + b"body { color: white; padding: 32px; background-color: #1b1b1b; }" * 20
                    + b"</style></head><body>DroidCam is Busy</body></html>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/video"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=droidcam")
            self.end_headers()
            try:
                for i in range(600):
                    scrambled = self.server.scramble and not (self.server.clean_640 and "640x480" in self.path)
                    jpg = _scrambled_jpeg() if scrambled else _jpeg(200 + i % 40)
                    self.wfile.write(b"--droidcam\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                                     % len(jpg) + jpg + b"\r\n")
                    self.wfile.flush()
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            return
        if self.path.startswith("/photoaf.jpg") and self.server.photo:  # IP Webcam full-resolution photo
            body = _jpeg(230, size=(480, 640))
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture()
def fake_phone():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeDroidCam)
    server.daemon_threads = True
    server.busy = server.scramble = server.photo = server.clean_640 = False
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def test_connect_preview_and_capture(fake_phone):
    stream = PhoneStream()
    status = stream.connect("127.0.0.1", fake_phone.server_address[1])
    assert status["connected"] and (status["width"], status["height"]) == (160, 120)
    assert next(stream.frames())[:2] == b"\xff\xd8"  # the live preview gets JPEG frames
    jpg = stream.capture(settle=0.3)
    assert cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR).shape == (120, 160, 3)
    stream.stop()
    assert not stream.connected


def test_scrambled_droidcam_picture_is_flagged_and_not_captured(fake_phone):
    fake_phone.scramble = True
    stream = PhoneStream()
    status = stream.connect("127.0.0.1", fake_phone.server_address[1])
    assert status["scrambled"] and "IP Webcam" in status["warning"]
    with pytest.raises(CameraError, match="scrambled"):
        stream.capture(settle=0.2)
    stream.stop()


def test_ip_webcam_capture_uses_the_full_resolution_photo(fake_phone):
    fake_phone.photo = True
    stream = PhoneStream()
    stream.connect("127.0.0.1", fake_phone.server_address[1])
    jpg = stream.capture(settle=0.2)
    assert cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (480, 640)
    stream.stop()


def test_droidcam_force_takes_over_a_busy_phone(fake_phone, monkeypatch):
    port = fake_phone.server_address[1]
    monkeypatch.setattr(camera, "DROIDCAM_PORTS", {port})  # treat the fake server as DroidCam
    fake_phone.busy = True  # plain /video is busy; /video/force/... works (as in DroidCam's help)
    stream = PhoneStream()
    status = stream.connect("127.0.0.1", port, wait=3)
    assert status["connected"] and "/video/force/" in stream.url
    stream.stop()


def test_scrambled_droidcam_hd_falls_back_to_640x480(fake_phone, monkeypatch):
    port = fake_phone.server_address[1]
    monkeypatch.setattr(camera, "DROIDCAM_PORTS", {port})
    fake_phone.scramble = fake_phone.clean_640 = True  # HD scrambled, 640x480 clean
    stream = PhoneStream()
    status = stream.connect("127.0.0.1", port, wait=3)
    assert not status["scrambled"] and stream.url.endswith("640x480")
    stream.stop()


class _FakePCCamera:
    """Stands in for cv2.VideoCapture: only camera 1 exists (like 'DroidCam Video' on a PC without webcam)."""

    def __init__(self, index, backend=None):
        self.index = index

    def isOpened(self):
        return self.index == 1

    def set(self, *args):
        return True

    def read(self):
        time.sleep(0.02)
        return True, np.full((480, 640, 3), 220, np.uint8)

    def release(self):
        pass


class _NoPCCamera(_FakePCCamera):
    def isOpened(self):
        return False


def test_droidcam_pc_client_camera_is_found_and_captured(monkeypatch):
    monkeypatch.setattr(camera.cv2, "VideoCapture", _FakePCCamera)
    monkeypatch.setattr(camera, "droidcam_client_running", lambda: True)
    stream = PhoneStream()
    status = stream.connect_pc()  # "Auto": tries cameras 0, 1, ... and keeps the first with pictures
    assert status["connected"] and status["source"] == "pc" and status["app"] == "DroidCam (PC client)"
    assert (status["width"], status["height"]) == (640, 480) and stream.port == 1
    assert stream.capture(settle=0.2)[:2] == b"\xff\xd8"
    stream.stop()


def test_connect_uses_the_droidcam_pc_client_when_it_runs(monkeypatch):
    import app as app_module

    monkeypatch.setattr(camera.cv2, "VideoCapture", _FakePCCamera)
    monkeypatch.setattr(camera, "droidcam_client_running", lambda: True)
    monkeypatch.setattr(app_module, "droidcam_client_running", lambda: True)
    client = login(app_module.app.test_client())
    data = client.post("/camera/phone/connect", data={"host": "192.168.1.5", "port": 4747}).get_json()
    assert data["ok"] and data["source"] == "pc"  # no direct phone connection, so no "busy"
    client.post("/camera/phone/disconnect")


def test_no_pc_camera_gives_a_clear_message(monkeypatch):
    monkeypatch.setattr(camera.cv2, "VideoCapture", _NoPCCamera)
    monkeypatch.setattr(camera, "droidcam_client_running", lambda: False)
    with pytest.raises(CameraError, match="No camera on this PC"):
        PhoneStream().connect_pc(wait=1)


def test_busy_phone_gives_a_clear_message(fake_phone):
    fake_phone.busy = True
    with pytest.raises(CameraError, match="busy: another app or computer"):
        PhoneStream().connect("127.0.0.1", fake_phone.server_address[1], wait=3, busy_retries=1)


def test_closed_port_explains_what_to_do():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # nothing listens here once the socket is closed
    with pytest.raises(CameraError, match="connect"):
        PhoneStream().connect("127.0.0.1", port, wait=3)


def test_internet_addresses_are_refused():
    with pytest.raises(CameraError, match="local network"):
        PhoneStream().connect("8.8.8.8", 4747)


def test_web_connect_preview_capture(fake_phone):
    import app as app_module
    from invoice_parser.matcher import LabelMatcher
    from invoice_parser.pipeline import InvoiceParser

    for w in app_module.workers.values():
        w._parser = InvoiceParser(matcher=LabelMatcher(use_embeddings=False))
    client = login(app_module.app.test_client())
    resp = client.post("/camera/phone/connect", data={"host": "127.0.0.1", "port": fake_phone.server_address[1]})
    assert resp.get_json()["ok"]

    preview = client.get("/camera/phone/stream", buffered=False)
    assert b"image/jpeg" in next(iter(preview.response))
    preview.close()

    capture = client.post("/camera/phone/capture")
    assert capture.status_code == 302 and "/invoices/" in capture.headers["Location"]
    assert client.get("/camera").status_code == 200  # the address is remembered on the page
    client.post("/camera/phone/disconnect")
    assert not app_module.phone.connected
