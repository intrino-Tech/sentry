"""
camera_helper.py
================
Camera helpers for the fire/smoke detection system.

Three things live here:

1. discover_camera()  -> auto-detect a working LOCAL camera (USB/integrated)
                         using blank-frame rejection. For prototyping on one PC.

2. open_source()      -> open EITHER a local index OR an RTSP/HTTP URL and
                         return (cap, first_frame) once a valid frame is seen.
                         For RTSP: forces TCP transport + low-latency FFmpeg
                         flags to minimise end-to-end delay.

3. CameraStream       -> a threaded reader for ONE source (local or network)
                         with auto-reconnect + watchdog. This is what you use
                         in the real multi-camera pipeline, where each camera
                         runs in its own thread and feeds (optionally batched)
                         inference.

Local cameras use DirectShow on Windows; network streams use FFMPEG with
low-latency flags (nobuffer, low_delay, reorder_queue_size=0) to cut the
default 1-3s RTSP buffer delay down to ~200-300ms.
"""

import os
import cv2
import time
import threading
import numpy as np


# ==========================
# Config
# ==========================
WIDTH = 640
HEIGHT = 480

# Blank-frame thresholds. A real frame has spatial variation (std) and isn't
# pitch black (mean). Lower STD_MIN if cameras sit in genuinely dark zones.
STD_MIN = 12
MEAN_MIN = 10

# Low-latency FFmpeg options for RTSP streams.
# - rtsp_transport=tcp   : avoids UDP packet loss / reordering on busy networks
# - buffer_size          : small network receive buffer
# - max_delay=0          : no deliberate delay added by the demuxer
# - reorder_queue_size=0 : don't buffer packets to reorder them
# - fflags=nobuffer      : skip the input buffer entirely
# - flags=low_delay      : tell the codec to minimise latency
RTSP_OPTIONS = (
    "rtsp_transport;tcp|"
    "buffer_size;1024000|"
    "max_delay;0|"
    "reorder_queue_size;0|"
    "fflags;nobuffer|"
    "flags;low_delay"
)


# ==========================
# Helper: validate a frame
# ==========================
def _is_valid_frame(frame):
    """True if the frame looks like real content, not a blank/solid/black frame."""
    if frame is None:
        return False
    return np.std(frame) > STD_MIN and np.mean(frame) > MEAN_MIN


# ==========================
# 1. Local camera auto-detect
# ==========================
def discover_camera(camera_order=None, use_msmf=False, warmup_frames=20):
    """
    Scan LOCAL camera indices/backends and return the first one that yields a
    valid (non-blank) frame.

    Returns (cap, index, backend_name) on success, or (None, None, None).

    Note: MSMF is skipped by default because on some machines it stalls ~20s
    per index. Set use_msmf=True only if DirectShow can't find your camera.
    """
    if camera_order is None:
        camera_order = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]  # index 1 first (external)

    backends = [("DirectShow", cv2.CAP_DSHOW), ("Default", cv2.CAP_ANY)]
    if use_msmf:
        backends.append(("MediaFoundation", cv2.CAP_MSMF))

    print("\n===================================")
    print(" Searching for Available Cameras")
    print("===================================")

    for backend_name, backend in backends:
        print(f"\nBackend : {backend_name}")
        for cam_id in camera_order:
            print(f"  Checking camera {cam_id}...", end=" ")
            camera = cv2.VideoCapture(cam_id, backend)
            if not camera.isOpened():
                print("cannot open.")
                camera.release()
                continue

            camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            time.sleep(0.5)  # let the sensor warm up

            best_frame = None
            for _ in range(warmup_frames):
                ret, frame = camera.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue
                if _is_valid_frame(frame):
                    best_frame = frame
                    break
                time.sleep(0.05)

            if best_frame is not None:
                print("VALID.")
                print("\n===================================")
                print(" VALID CAMERA FOUND")
                print("===================================")
                print(f"Backend    : {backend_name}")
                print(f"Camera ID  : {cam_id}")
                print(f"Resolution : {best_frame.shape[1]} x {best_frame.shape[0]}")
                print(f"Mean Pixel : {np.mean(best_frame):.2f}")
                print(f"Std Dev    : {np.std(best_frame):.2f}")
                print("===================================\n")
                return camera, cam_id, backend_name

            print("invalid/blank.")
            camera.release()

    print("\nNo valid camera found.\n")
    return None, None, None


# ==========================
# 2. Open a single source (local index OR RTSP/HTTP URL)
# ==========================
def open_source(source, backend=cv2.CAP_DSHOW, warmup_frames=20,
                open_timeout_ms=5000, read_timeout_ms=5000):
    """
    Open EITHER a local camera index (int) OR a network URL (str), e.g.
    'rtsp://192.168.88.8:554/video/live?channel=1&subtype=1'.

    For RTSP/HTTP URLs:
      - Forces TCP transport to avoid UDP packet loss on busy LAN switches.
      - Sets low-latency FFmpeg flags (nobuffer, low_delay) to cut the
        default 1-3s decode buffer to ~200-300ms.
      - Sets CAP_PROP_BUFFERSIZE=1 so the reader always gets the latest
        frame, not a backlog.

    Returns (cap, first_valid_frame) on success, or (None, None).
    """
    is_url = isinstance(source, str)

    if is_url:
        # Apply low-latency FFmpeg options BEFORE opening the capture.
        # These are picked up by OpenCV's FFmpeg backend at open time.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = RTSP_OPTIONS
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        # Fail fast instead of blocking forever on a dead camera.
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout_ms)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout_ms)
        except Exception:
            pass  # property not supported on all OpenCV builds
    else:
        cap = cv2.VideoCapture(source, backend)

    if not cap.isOpened():
        print(f"[open_source] cannot open: {source}")
        cap.release()
        return None, None

    if is_url:
        # Buffer size 1 => always decode the LATEST frame, never a backlog.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    time.sleep(0.5)

    for _ in range(warmup_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue
        if _is_valid_frame(frame):
            print(f"[open_source] valid: {source} ({frame.shape[1]}x{frame.shape[0]})")
            return cap, frame
        time.sleep(0.05)

    print(f"[open_source] opened but no valid frame: {source}")
    cap.release()
    return None, None


# ==========================
# 3. Threaded camera with auto-reconnect + watchdog
# ==========================
class CameraStream:
    """
    Background reader for ONE source (local index or RTSP URL).

    - Reads frames in a dedicated thread, always keeping the latest frame.
    - Auto-reconnects if the feed drops or goes stale (frozen RTSP).
    - Exposes .read() for the main/inference loop and .healthy() for monitoring.

    Usage:
        stream = CameraStream("rtsp://192.168.88.8:554/video/live?channel=1&subtype=1",
                              name="Weld Bay")
        stream.start()
        while True:
            ok, frame = stream.read()
            if ok:
                results = model(frame, ...)
            ...
        stream.stop()
    """

    def __init__(self, source, name=None, backend=cv2.CAP_DSHOW,
                 reconnect_delay=2.0, stale_timeout=5.0):
        self.source = source
        self.name = name or f"cam:{source}"
        self.backend = backend
        self.reconnect_delay = reconnect_delay   # wait before re-opening
        self.stale_timeout = stale_timeout       # no fresh frame => unhealthy

        self.cap = None
        self.frame = None
        self.last_frame_time = 0.0
        self.connected = False

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    # ---- public API ----
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def read(self):
        """Return (ok, frame) with the most recent frame (copy)."""
        with self._lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def healthy(self):
        """True if connected and a fresh frame arrived within stale_timeout."""
        return self.connected and (time.time() - self.last_frame_time) < self.stale_timeout

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._release()

    # ---- internals ----
    def _open(self):
        cap, frame = open_source(self.source, backend=self.backend)
        if cap is None:
            return False
        self.cap = cap
        with self._lock:
            self.frame = frame
            self.last_frame_time = time.time()
        self.connected = True
        print(f"[{self.name}] connected.")
        return True

    def _release(self):
        self.connected = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _run(self):
        while not self._stop.is_set():
            # (Re)connect if needed
            if self.cap is None:
                if not self._open():
                    print(f"[{self.name}] connect failed, retrying in {self.reconnect_delay}s")
                    time.sleep(self.reconnect_delay)
                    continue

            ret, frame = self.cap.read()

            # Drop or freeze handling
            if not ret or frame is None:
                print(f"[{self.name}] read failed -> reconnecting")
                self._release()
                time.sleep(self.reconnect_delay)
                continue

            with self._lock:
                self.frame = frame
                self.last_frame_time = time.time()

            # Watchdog: a frozen RTSP feed keeps returning the SAME stale frame.
            # If frames stop being fresh, force a reconnect.
            if not self.healthy():
                print(f"[{self.name}] stale feed -> reconnecting")
                self._release()
                time.sleep(self.reconnect_delay)

        self._release()
        print(f"[{self.name}] stopped.")


# ==========================
# Quick self-test
# ==========================
if __name__ == "__main__":
    import sys

    # Pass an RTSP URL as an argument to test the network path:
    #   python camera_helper.py rtsp://192.168.88.8:554/video/live?channel=1&subtype=1
    # With no argument, it auto-detects a local camera.
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        cap, idx, backend = discover_camera()
        if cap is None:
            print("No camera; exiting.")
            sys.exit(1)
        cap.release()       # discover_camera opened it; CameraStream reopens
        src = idx

    stream = CameraStream(src, name="test").start()
    print("Showing stream. Press 'q' to quit.")

    try:
        while True:
            ok, frame = stream.read()
            if ok:
                status = "OK" if stream.healthy() else "STALE"
                cv2.putText(frame, f"{stream.name} [{status}]", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()