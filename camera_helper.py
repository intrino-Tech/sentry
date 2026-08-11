"""
camera_helper.py
================
Camera helpers for the fire/smoke detection system.

Three things live here:

1. discover_camera()  -> auto-detect a working LOCAL camera (USB/integrated)
                         using blank-frame rejection. For prototyping.

2. open_source()      -> open EITHER a local index OR an RTSP/HTTP URL and
                         return (cap, first_frame) once a valid frame is seen.
                         On Jetson, prefers GStreamer + NVDEC hardware decode
                         when available, falling back to FFmpeg.

3. CameraStream       -> a threaded reader for ONE source (local or network)
                         with auto-reconnect + a real freeze watchdog.

--------------------------------------------------------------------------
CHANGES FROM THE PREVIOUS VERSION
--------------------------------------------------------------------------
* RTSP_OPTIONS no longer sets `fflags;nobuffer` or `reorder_queue_size;0`.
  Those two flags were the cause of the continuous
      "Could not find ref with POC N / Error constructing the frame RPS"
  errors. HEVC legitimately delivers packets out of decode order because of
  B-frames and multi-frame reference chains. With zero reorder capacity the
  demuxer handed the decoder packets whose reference frames had not arrived
  yet, so every frame failed to resolve its reference picture set. The steady
  march of POC numbers (21, 22, 23...) was the giveaway: genuine packet loss
  is bursty and irregular, not perfectly sequential. A small reorder queue
  costs ~100-200 ms of latency, which is irrelevant for fire detection.

* The freeze watchdog now actually works. Previously `healthy()` was called
  immediately after `last_frame_time` was refreshed, so it could never return
  False. A frozen RTSP feed that keeps re-delivering the same buffer would
  never trigger a reconnect. Freshness is now judged on frame CONTENT.

* Hardware decode via `nvv4l2decoder` is used when OpenCV was built with
  GStreamer. This moves HEVC decoding off the CPU onto the Orin's dedicated
  NVDEC block, freeing cores for inference and the Flask app.

* Default backend is CAP_ANY, not CAP_DSHOW (which is Windows-only).
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
# pitch black (mean). These are deliberately permissive: a fire camera must
# keep working in a dark room at night, and an over-strict threshold here
# causes an endless reconnect loop after dusk. Raise only if you are getting
# false "valid" results from a disconnected sensor.
STD_MIN = 6
MEAN_MIN = 4

# FFmpeg options for RTSP streams.
#   rtsp_transport=tcp    : retransmit lost packets instead of leaving holes
#                           in the reference chain
#   stimeout=5000000      : 5 s socket timeout (microseconds); also stops
#                           cap.read() blocking forever at shutdown
#   buffer_size           : network receive buffer
#   max_delay=500000      : 0.5 s of demuxer reordering headroom
#   reorder_queue_size=10 : hold a few packets so out-of-order HEVC packets
#                           can be put back in decode order
#   flags=low_delay       : still ask the codec to minimise latency
RTSP_OPTIONS = (
    "rtsp_transport;tcp|"
    "stimeout;5000000|"
    "buffer_size;1024000|"
    "max_delay;500000|"
    "reorder_queue_size;10|"
    "flags;low_delay"
)

# Prefer hardware decode when OpenCV supports GStreamer. Set the env var
# SENTRY_FORCE_FFMPEG=1 to disable and fall back to FFmpeg.
USE_GSTREAMER = (
    "GStreamer:                   YES" in cv2.getBuildInformation()
    and os.environ.get("SENTRY_FORCE_FFMPEG", "0") != "1"
)

# How many consecutive identical frames before declaring the feed frozen.
# At ~15 fps, 60 frames is roughly 4 seconds of a motionless picture.
FREEZE_FRAME_LIMIT = 60


# ==========================
# Helpers
# ==========================
def _is_valid_frame(frame):
    """True if the frame looks like real content, not a blank/solid/black frame."""
    if frame is None:
        return False
    return np.std(frame) > STD_MIN and np.mean(frame) > MEAN_MIN


def _frame_signature(frame):
    """
    Cheap content fingerprint used for freeze detection.

    Subsampling every 16th pixel keeps this near-free even at 1080p while
    still changing on any real motion or sensor noise. A truly frozen feed
    re-delivers a byte-identical buffer, so the signature stops changing.
    """
    return hash(frame[::16, ::16].tobytes())


def _gst_pipeline(source, codec="h265", latency=200):
    """
    Build a GStreamer pipeline that decodes on the Orin's NVDEC block.

    `drop=true max-buffers=1` on the appsink means the reader always gets the
    newest frame rather than working through a backlog, which is what you want
    for live detection: a stale frame is worse than a skipped one.
    """
    depay = "rtph265depay ! h265parse" if codec == "h265" else "rtph264depay ! h264parse"
    return (
        f"rtspsrc location={source} protocols=tcp latency={latency} ! "
        f"{depay} ! nvv4l2decoder ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


# ==========================
# 1. Local camera auto-detect
# ==========================
def discover_camera(camera_order=None, backends=None, warmup_frames=20):
    """
    Scan LOCAL camera indices/backends and return the first one that yields a
    valid (non-blank) frame.

    Returns (cap, index, backend_name) on success, or (None, None, None).
    """
    if camera_order is None:
        camera_order = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]  # index 1 first (external)

    if backends is None:
        if os.name == "nt":
            backends = [("DirectShow", cv2.CAP_DSHOW), ("Default", cv2.CAP_ANY)]
        else:
            backends = [("V4L2", cv2.CAP_V4L2), ("Default", cv2.CAP_ANY)]

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
def open_source(source, backend=cv2.CAP_ANY, warmup_frames=20,
                open_timeout_ms=5000, read_timeout_ms=5000,
                codec="h265", prefer_hw=True):
    """
    Open EITHER a local camera index (int) OR a network URL (str).

    Network sources try GStreamer + NVDEC first (hardware decode on Jetson),
    then fall back to FFmpeg with TCP transport and a small reorder queue.

    Returns (cap, first_valid_frame) on success, or (None, None).
    """
    is_url = isinstance(source, str)
    cap = None

    if is_url and prefer_hw and USE_GSTREAMER:
        pipeline = _gst_pipeline(source, codec=codec)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"[open_source] using GStreamer/NVDEC for {source}")
        else:
            print("[open_source] GStreamer pipeline failed, falling back to FFmpeg")
            cap.release()
            cap = None

    if cap is None and is_url:
        # Must be set BEFORE the VideoCapture is constructed; OpenCV reads it
        # at open time and ignores later changes.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = RTSP_OPTIONS
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        # Fail fast instead of blocking forever on a dead camera.
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout_ms)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout_ms)
        except Exception:
            pass  # property not supported on all OpenCV builds

    if cap is None:
        cap = cv2.VideoCapture(source, backend)

    if not cap.isOpened():
        print(f"[open_source] cannot open: {source}")
        cap.release()
        return None, None

    if is_url:
        # Buffer size 1 => always decode the LATEST frame, never a backlog.
        # (No-op on the GStreamer path, where the appsink already does this.)
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
# 3. Threaded camera with auto-reconnect + freeze watchdog
# ==========================
class CameraStream:
    """
    Background reader for ONE source (local index or RTSP URL).

    - Reads frames in a daemon thread, always keeping the latest frame.
    - Auto-reconnects if the feed drops.
    - Detects a FROZEN feed (same picture re-delivered) and reconnects.

    Usage:
        stream = CameraStream("rtsp://.../live", name="Weld Bay").start()
        while True:
            ok, frame = stream.read()
            if ok:
                results = model(frame, ...)
        stream.stop()
    """

    def __init__(self, source, name=None, backend=cv2.CAP_ANY,
                 reconnect_delay=2.0, stale_timeout=5.0,
                 codec="h265", prefer_hw=True):
        self.source = source
        self.name = name or f"cam:{source}"
        self.backend = backend
        self.reconnect_delay = reconnect_delay   # wait before re-opening
        self.stale_timeout = stale_timeout       # no NEW content => unhealthy
        self.codec = codec
        self.prefer_hw = prefer_hw

        self.cap = None
        self.frame = None
        self.last_frame_time = 0.0
        self.last_change_time = 0.0              # last time content CHANGED
        self.connected = False
        self.frames_read = 0

        self._prev_sig = None
        self._identical = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    # ---- public API ----
    def start(self):
        # daemon=True matters: a non-daemon thread blocked inside a native
        # socket read cannot be joined at interpreter shutdown, which is what
        # produced "FATAL: exception not rethrown / Aborted (core dumped)".
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
        """True if connected and the picture has CHANGED within stale_timeout."""
        if not self.connected:
            return False
        return (time.time() - self.last_change_time) < self.stale_timeout

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._release()

    # ---- internals ----
    def _open(self):
        cap, frame = open_source(self.source, backend=self.backend,
                                 codec=self.codec, prefer_hw=self.prefer_hw)
        if cap is None:
            return False
        self.cap = cap
        now = time.time()
        with self._lock:
            self.frame = frame
            self.last_frame_time = now
            self.last_change_time = now
        self._prev_sig = _frame_signature(frame)
        self._identical = 0
        self.connected = True
        print(f"[{self.name}] connected.")
        return True

    def _release(self):
        self.connected = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _reconnect(self, reason):
        print(f"[{self.name}] {reason} -> reconnecting")
        self._release()
        self._prev_sig = None
        self._identical = 0
        # Interruptible sleep so Ctrl+C doesn't wait out the full delay.
        self._stop.wait(self.reconnect_delay)

    def _run(self):
        while not self._stop.is_set():
            # (Re)connect if needed
            if self.cap is None:
                if not self._open():
                    print(f"[{self.name}] connect failed, retrying in {self.reconnect_delay}s")
                    self._stop.wait(self.reconnect_delay)
                    continue

            ret, frame = self.cap.read()

            # Drop handling
            if not ret or frame is None:
                self._reconnect("read failed")
                continue

            now = time.time()
            sig = _frame_signature(frame)

            # Freeze detection. A frozen RTSP feed keeps returning a
            # byte-identical buffer while cap.read() still reports success,
            # so the only reliable signal is that the CONTENT stops changing.
            if sig == self._prev_sig:
                self._identical += 1
            else:
                self._identical = 0
                self.last_change_time = now
            self._prev_sig = sig

            with self._lock:
                self.frame = frame
                self.last_frame_time = now
                self.frames_read += 1

            if self._identical > FREEZE_FRAME_LIMIT:
                self._reconnect(f"frozen feed ({self._identical} identical frames)")
                continue

        self._release()
        print(f"[{self.name}] stopped.")


# ==========================
# Quick self-test
# ==========================
if __name__ == "__main__":
    import sys

    # Pass an RTSP URL as an argument to test the network path:
    #   python3 camera_helper.py "rtsp://user:pass@192.168.88.8:554/video/live?channel=1&subtype=2"
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

    print(f"GStreamer available: {USE_GSTREAMER}")
    stream = CameraStream(src, name="test").start()
    print("Reading stream. Press Ctrl+C to quit.")

    headless = os.environ.get("DISPLAY") is None and os.name != "nt"
    t0 = time.time()

    try:
        while True:
            ok, frame = stream.read()
            if ok and not headless:
                status = "OK" if stream.healthy() else "STALE"
                cv2.putText(frame, f"{stream.name} [{status}]", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("test", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif headless:
                time.sleep(1.0)
                elapsed = time.time() - t0
                fps = stream.frames_read / elapsed if elapsed > 0 else 0
                print(f"frames={stream.frames_read}  fps={fps:.1f}  healthy={stream.healthy()}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        stream.stop()
        if not headless:
            cv2.destroyAllWindows()