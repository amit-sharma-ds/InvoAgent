"""Get a picture of an invoice from a phone camera app over Wi-Fi, or from a webcam on this PC.

Phone apps (no PC software needed):
  * DroidCam (port 4747) and IP Webcam (port 8080) both serve an MJPEG video stream at /video.
  * PhoneStream keeps ONE connection to the phone and shares it between the live preview in
    the browser and the capture button. DroidCam accepts only one viewer at a time, so the
    browser never connects to the phone directly.
  * On capture the phone is asked to autofocus and the sharpest of the latest frames is used.
  * The connection is closed after IDLE_SECONDS without a viewer, so the phone is free again.
"""
import ipaddress
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import config


class CameraError(RuntimeError):
    pass


PHONE_APPS = {4747: "DroidCam", 8080: "IP Webcam"}
DROIDCAM_PORTS = {4747}
SETTINGS_PATH = config.DATA_DIR / "settings.json"


def droidcam_client_running() -> bool:
    """True when the DroidCam Windows client runs on this PC. The client receives the phone's
    normal picture and offers it as the 'DroidCam Video' camera."""
    if not sys.platform.startswith("win"):
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq droidcam.exe", "/NH"], capture_output=True,
                             text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "droidcam.exe" in out.lower()


# --- remembered phone address --------------------------------------------------------------
def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(**values):
    data = load_settings()
    data.update(values)
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, SETTINGS_PATH)


# --- address checks --------------------------------------------------------------------------
def _require_lan(host: str):
    """Only phones on the local network: the server must never be pointed at the internet."""
    if not host:
        raise CameraError("Enter the phone's IP address, for example 192.168.1.5")
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None, socket.AF_INET)}
    except OSError as exc:
        raise CameraError(f"Unknown address '{host}'.") from exc
    for addr in addresses:
        ip = ipaddress.ip_address(addr)
        if not (ip.is_private or ip.is_loopback):
            raise CameraError("Only phones on the local network (for example 192.168.x.x) are allowed.")


def phone_video_url(host: str, port: int) -> str:
    host = (host or "").strip()
    _require_lan(host)
    if not 1 <= int(port) <= 65535:
        raise CameraError("The port must be a number between 1 and 65535 (DroidCam uses 4747).")
    return f"http://{host}:{int(port)}/video"


def ip_camera_snapshot_url(base_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise CameraError("Enter a full address such as http://192.168.1.20:8080")
    try:  # only local-network IPs; host names are allowed (phone names on the network)
        ip = ipaddress.ip_address(parsed.hostname)
        if not (ip.is_private or ip.is_loopback):
            raise CameraError("Only local-network (LAN) phone addresses are allowed.")
    except ValueError:
        pass
    if url.lower().endswith((".jpg", ".jpeg")):
        return url
    if url.lower().endswith("/video"):
        url = url[: -len("/video")]
    return url + "/shot.jpg"


def fetch_ip_camera_frame(base_url: str, timeout: float = 10.0) -> bytes:
    """One snapshot from IP Webcam's /shot.jpg (kept for the JSON API)."""
    url = ip_camera_snapshot_url(base_url)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read(25 * 1024 * 1024)
    except OSError as exc:
        raise CameraError(f"Could not reach the phone camera at {url}. ({exc})") from exc
    if not data:
        raise CameraError("The phone camera returned an empty image.")
    return data


# --- live phone stream -----------------------------------------------------------------------
def _connect_problem(host, port, exc) -> str:
    """Plain-language reason why the phone could not be reached."""
    app = PHONE_APPS.get(port, "camera")
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, ConnectionRefusedError):
        return (f"Could not connect: the phone {host} is on the network, but nothing answers on port {port}. "
                f"Open the {app} app on the phone (its screen shows the Wi-Fi IP and port) and press Connect again.")
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return (f"Could not connect: no answer from {host}:{port}. Check the IP shown in the {app} app and that "
                "the phone and this PC are on the same Wi-Fi.")
    return f"Could not connect to the phone at {host}:{port} ({reason})."


BUSY_MESSAGE = (
    "DroidCam says it is busy: another app or computer is already watching this phone (for example the DroidCam "
    "Windows client, OBS, or a browser tab with the phone's address). Close that, or stop and start DroidCam on "
    "the phone, then press Connect again.")


def _page_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


SCRAMBLED_MESSAGE = (
    "DroidCam is sending a scrambled picture (green lines with 'using droidcam.app'), so the invoice cannot be "
    "read. Keep the DroidCam app open on the phone screen (Android blocks the camera for apps in the background). "
    "DroidCam's own help says the overlay on its browser / IP-camera video can only be changed or removed in the "
    "paid DroidCamX. Other options: the free 'IP Webcam' app (port 8080) with this same Connect button, "
    "DroidCam's Windows client with 'Webcam on this PC', or option 1, a photo from the phone's browser.")


def _is_scrambled(jpg: bytes) -> bool:
    """True for frames made of horizontal scan lines (DroidCam's obscured feed). Normal photos and
    documents score about 0.9 on this ratio; the scrambled DroidCam frame scored 37."""
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    img = img.astype(np.float32)
    vertical = np.abs(np.diff(img, axis=0)).mean()
    horizontal = np.abs(np.diff(img, axis=1)).mean()
    return bool(vertical / max(horizontal, 1e-6) > 4.0)


def _sharpness(jpg: bytes) -> float:
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1.0
    if img.shape[1] > 800:
        img = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


class PhoneStream:
    IDLE_SECONDS = 60

    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._frames = deque(maxlen=10)  # (time received, jpeg bytes)
        self._thread = None
        self._stop = threading.Event()
        self._last_used = 0.0
        self.url = None
        self.base = None  # http://host:port
        self.host = None
        self.port = None
        self.source = None  # "wifi" (phone app over the network) or "pc" (camera on this PC)
        self.label = None
        self.error = None
        self.frame_count = 0

    @property
    def connected(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> dict:
        width = height = None
        with self._cond:
            last = self._frames[-1][1] if self._frames else None
        if last:
            img = cv2.imdecode(np.frombuffer(last, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                height, width = img.shape[:2]
        scrambled = bool(last) and _is_scrambled(last)
        app = self.label if self.source == "pc" else PHONE_APPS.get(self.port, "phone camera")
        return {"connected": self.connected and last is not None, "host": self.host, "port": self.port,
                "frames": self.frame_count, "width": width, "height": height, "error": self.error,
                "app": app, "source": self.source, "scrambled": scrambled,
                "warning": SCRAMBLED_MESSAGE if scrambled else None}

    # --- camera on this PC (webcam, or the DroidCam client's 'DroidCam Video' camera) ---------
    def connect_pc(self, index=None, wait: float = 8.0) -> dict:
        """Open a camera on this PC and keep it streaming (live preview + capture).
        With index=None the first camera that delivers pictures is used."""
        label = "DroidCam (PC client)" if droidcam_client_running() else "PC camera"
        for idx in ([int(index)] if index is not None else range(6)):
            with self._lock:
                self._stop_thread()
                self.url, self.base, self.host, self.port = f"pc-camera:{idx}", None, "this PC", idx
                self.source, self.label, self.error, self.frame_count = "pc", label, None, 0
                self._frames.clear()
                self._stop = threading.Event()
                self._touch()
                self._thread = threading.Thread(target=self._run_pc, args=(idx, self._stop),
                                                name="pc-camera", daemon=True)
                self._thread.start()
                thread = self._thread
            deadline = time.time() + wait
            with self._cond:
                while not self._frames and thread.is_alive() and time.time() < deadline:
                    self._cond.wait(timeout=0.2)
            if self._frames:
                return self.status()
        self.stop()
        if label.startswith("DroidCam"):
            raise CameraError("The DroidCam PC client is running, but its camera gives no picture yet. In the "
                              "DroidCam window on this PC, connect to the phone (Wi-Fi IP and port from the phone) "
                              "until you see the video there, then press Connect again.")
        raise CameraError("No camera on this PC gives a picture. For a phone, open the DroidCam app on the phone "
                          "and connect it in the DroidCam PC client, or use 'Phone camera over Wi-Fi'.")

    def _run_pc(self, index, stop):
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        try:
            if not cap.isOpened():
                raise CameraError(f"Camera {index} is not available on this PC.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # ask for HD; the camera gives the closest it supports
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            failures = 0
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures > 60:
                        raise CameraError(f"Camera {index} stopped sending pictures.")
                    time.sleep(0.05)
                    continue
                failures = 0
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if ok:
                    self._push(buf.tobytes())
                if time.time() - self._last_used > self.IDLE_SECONDS:
                    break  # nobody is watching: release the camera
        except CameraError as exc:
            self.error = str(exc)
        finally:
            cap.release()
            with self._cond:
                self._cond.notify_all()

    def connect(self, host: str, port: int, wait: float = 8.0, busy_retries: int = 2) -> dict:
        port = int(port)
        url = phone_video_url(host, port)
        base = url[: -len("/video")]
        with self._lock:
            if self.connected and self.base == base and self._frames:
                self._touch()
                return self.status()
        droidcam = port in DROIDCAM_PORTS
        force_url = base + "/video/force/1280x720"  # DroidCam help: "force" overrides existing connections
        plan = [url] + ([force_url] if droidcam else [url]) * busy_retries
        for i, attempt_url in enumerate(plan):
            if i > 1 or (i == 1 and not droidcam):
                time.sleep(2)  # a viewer that just left can keep the phone "busy" for a moment
            if self._try(attempt_url, host, port, wait) or self.error != BUSY_MESSAGE:
                break
        if self._frames and droidcam and self.status()["scrambled"]:
            # Some DroidCam versions obscure HD video for browser viewers: try the standard 640x480.
            if not self._try(base + "/video/force/640x480", host, port, wait) or self.status()["scrambled"]:
                self._try(force_url, host, port, wait)
        if not self._frames:
            self.stop()
            raise CameraError(self.error or (
                f"No picture from {host}:{port}. Open the {PHONE_APPS.get(port, 'camera')} app on the phone "
                "(its screen shows the Wi-Fi IP and port) and make sure phone and PC are on the same network."))
        return self.status()

    def _try(self, url, host, port, wait) -> bool:
        """Open one video URL and wait for the first frame."""
        thread = self._start(url, host, port)
        deadline = time.time() + wait
        with self._cond:
            while not self._frames and thread.is_alive() and time.time() < deadline:
                self._cond.wait(timeout=0.2)
        return bool(self._frames)

    def _start(self, url, host, port):
        with self._lock:
            self._stop_thread()
            self.url, self.host, self.port = url, host.strip(), int(port)
            self.base = f"http://{self.host}:{self.port}"
            self.source, self.label = "wifi", None
            self.error, self.frame_count = None, 0
            self._frames.clear()
            self._stop = threading.Event()
            self._touch()
            self._thread = threading.Thread(target=self._run, args=(url, self._stop),
                                            name="phone-camera", daemon=True)
            self._thread.start()
            return self._thread

    def stop(self):
        with self._lock:
            self._stop_thread()

    def _stop_thread(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)
        self._thread = None

    def _touch(self):
        self._last_used = time.time()

    def _open(self, url):
        last_error = None
        candidates = [url] + ([url[: -len("/video")] + "/mjpegfeed"] if url.endswith("/video") else [])
        for candidate in candidates:  # /mjpegfeed: older DroidCam versions
            try:
                return urllib.request.urlopen(candidate, timeout=8)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code != 404:
                    break
        raise CameraError(f"The phone app did not offer a video stream ({last_error}).")

    def _run(self, url, stop):
        resp = None
        try:
            try:
                resp = self._open(url)
            except OSError as exc:
                raise CameraError(_connect_problem(self.host, self.port, exc)) from exc
            content_type = resp.headers.get("Content-Type", "").lower()
            if "multipart" not in content_type and "image" not in content_type:
                text = _page_text(resp.read(16384).decode("utf-8", "ignore"))
                if "busy" in text.lower():
                    raise CameraError(BUSY_MESSAGE)
                raise CameraError(f"The phone answered with a page instead of video: {text[:150]!r}")
            buf = b""
            while not stop.is_set():
                chunk = resp.read1(65536)
                if not chunk:
                    raise CameraError("The phone stopped sending video.")
                buf += chunk
                while True:  # cut complete JPEG frames out of the MJPEG stream
                    start = buf.find(b"\xff\xd8")
                    if start < 0:
                        buf = buf[-1:]
                        break
                    end = buf.find(b"\xff\xd9", start + 2)
                    if end < 0:
                        buf = buf[start:]
                        break
                    self._push(buf[start:end + 2])
                    buf = buf[end + 2:]
                if len(buf) > 8 * 1024 * 1024:
                    buf = b""
                if time.time() - self._last_used > self.IDLE_SECONDS:
                    break  # nobody is watching: free the phone for other apps
        except CameraError as exc:
            self.error = str(exc)
        except OSError as exc:
            self.error = f"Lost the connection to the phone camera ({exc})."
        finally:
            if resp is not None:
                try:
                    resp.close()
                except OSError:
                    pass
            with self._cond:
                self._cond.notify_all()

    def _push(self, jpg: bytes):
        with self._cond:
            self._frames.append((time.time(), jpg))
            self.frame_count += 1
            self._cond.notify_all()

    def frames(self):
        """Yield each new JPEG frame while connected (used by the live preview)."""
        seen = -1
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self.frame_count != seen or not self.connected, timeout=5)
                if self.frame_count == seen or not self._frames:
                    if not self.connected:
                        return
                    continue
                seen = self.frame_count
                jpg = self._frames[-1][1]
            self._touch()
            yield jpg

    def _autofocus(self):
        if not self.base:
            return
        path = "/cam/1/af" if self.port in DROIDCAM_PORTS else "/focus"  # DroidCam / IP Webcam
        try:
            urllib.request.urlopen(self.base + path, timeout=1.5).close()
        except OSError:
            pass  # not every app version supports it

    def _full_photo(self):
        """IP Webcam can take a real photo (full camera resolution, autofocus): much better for OCR
        than a video frame. Returns JPEG bytes, or None when the app does not offer it."""
        if self.port in DROIDCAM_PORTS or not self.base:
            return None
        for path in ("/photoaf.jpg", "/photo.jpg"):
            try:
                with urllib.request.urlopen(self.base + path, timeout=15) as resp:
                    data = resp.read(30 * 1024 * 1024)
                if data[:2] == b"\xff\xd8":
                    return data
            except OSError:
                continue
        return None

    def capture(self, settle: float = 1.0) -> bytes:
        """Return the best picture as JPEG bytes: a full-resolution photo when the phone app
        offers one (IP Webcam), otherwise the sharpest recent video frame after autofocus."""
        if not self.connected:
            raise CameraError("The phone camera is not connected. Press Connect first.")
        self._touch()
        with self._cond:
            last = self._frames[-1][1] if self._frames else None
        if last and _is_scrambled(last):
            raise CameraError(SCRAMBLED_MESSAGE)
        photo = self._full_photo()
        if photo:
            return photo
        self._autofocus()
        started = time.time()
        time.sleep(settle)
        with self._cond:
            recent = [jpg for t, jpg in self._frames if t >= started] or [jpg for _, jpg in self._frames]
        if not recent:
            raise CameraError("No picture from the phone yet. Try again.")
        return max(recent, key=_sharpness)


# --- finding phones on the network -----------------------------------------------------------
def _own_lan_ip():
    """The PC's address on the local network (no packet is sent)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))
        return probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()


def find_phones(timeout: float = 0.35):
    """Look for DroidCam (4747) and IP Webcam (8080) on this PC's local /24 network."""
    own = _own_lan_ip()
    if not own or not ipaddress.ip_address(own).is_private:
        return []
    network = ipaddress.ip_network(f"{own}/24", strict=False)
    targets = [(str(h), p) for h in network.hosts() if str(h) != own for p in PHONE_APPS]

    def probe(target):
        try:
            with socket.create_connection(target, timeout=timeout) as sock:
                # Finish a normal request: a connection opened and dropped without one can leave
                # DroidCam showing "busy" for a while.
                sock.settimeout(1.0)
                sock.sendall(b"GET / HTTP/1.0\r\nHost: phone\r\n\r\n")
                try:
                    sock.recv(256)
                except OSError:
                    pass
                return target
        except OSError:
            return None

    with ThreadPoolExecutor(max_workers=64) as pool:
        open_ports = [t for t in pool.map(probe, targets) if t]

    phones = []
    for host, port in open_ports:
        if port == 8080:  # 8080 is common: keep it only if the page says it is IP Webcam
            try:
                with urllib.request.urlopen(f"http://{host}:{port}/", timeout=1.5) as resp:
                    if b"ip webcam" not in resp.read(4096).lower():
                        continue
            except OSError:
                continue
        phones.append({"host": host, "port": port, "app": PHONE_APPS[port]})
    return phones


# --- webcam on this PC -----------------------------------------------------------------------
def grab_local_camera(index: int = 0) -> bytes:
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise CameraError(f"No camera number {index} on this PC. For a phone, use "
                          "'Phone camera over Wi-Fi' above (no PC software needed).")
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        raise CameraError("Could not encode the camera picture.")
    return buf.tobytes()
