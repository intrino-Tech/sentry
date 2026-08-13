"""
config.py
=========
Shared settings for the whole system. Edit values here; both the detector
(algorithm) and the web UI read from this one place.

TENSORRT NOTE
-------------
The .engine files are compiled ON THIS DEVICE for THIS GPU (sm_87), against
THIS TensorRT version. They are NOT portable: you cannot build them on a
workstation and copy them over, you cannot copy them between projects with
different TensorRT versions, and they must be rebuilt after any JetPack
upgrade or after changing the power mode you run in (TensorRT picks kernels
by measured timing, so an engine profiled at 611 MHz may not be optimal at
full clocks).

Rebuild with:
    sudo nvpmodel -m 0 && sudo jetson_clocks     # profile at the clocks you run at
    YOLO_AUTOINSTALL=false yolo export model=firebest.pt format=engine half=True device=0
    YOLO_AUTOINSTALL=false yolo export model=best.pt     format=engine half=True device=0

A version mismatch surfaces as:
    AttributeError: 'NoneType' object has no attribute 'create_execution_context'
which looks like a code bug but is always a TensorRT version problem. Check
with:
    python3 -c "import tensorrt; print(tensorrt.__version__, tensorrt.__file__)"
The path must be under /usr/lib/python3.12/dist-packages (JetPack's copy). If
it points inside the venv, a pip wheel is shadowing JetPack's TensorRT and
every .engine will fail to load — uninstall it:
    pip uninstall -y tensorrt_cu13 tensorrt_cu13_bindings tensorrt_cu13_libs

Set USE_TENSORRT = False to fall back to the .pt files (slower, and depends on
PyTorch's CUDA kernels working on this GPU).
"""

import os
from pathlib import Path

# ---------------- .env loading ----------------
# Loaded HERE rather than in run.py so that every entry point — the service, a
# test one-liner, an export script — gets the same configuration without
# having to remember to load it first. Python executes top to bottom, so this
# must stay above the os.environ.get() calls further down.
_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            # .env files are NOT shell scripts: quotes are literal characters
            # unless stripped here, so CAM_PASS='x' would arrive as "'x'" and
            # the camera would reject it with 401 Unauthorized.
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ---------------- Runtime ----------------

USE_TENSORRT = True          # False -> use the .pt models instead

# Smoke detection is currently OFF. Running both a YOLO11s and an RT-DETR at
# 30 fps pinned the GPU at 98% with a single camera. Set True to re-enable;
# the smoke engine is only loaded when this is True, so turning it off also
# frees its device memory. Re-enabling requires best.engine to exist — export
# it first if you have only best.pt.
ENABLE_SMOKE = False

# ---------------- Models ----------------

FIRE_MODEL_PT = "firebest.pt"
SMOKE_MODEL_PT = "best.pt"

FIRE_MODEL_ENGINE = "firebest.engine"
SMOKE_MODEL_ENGINE = "best.engine"

FIRE_MODEL = FIRE_MODEL_ENGINE if USE_TENSORRT else FIRE_MODEL_PT
SMOKE_MODEL = SMOKE_MODEL_ENGINE if USE_TENSORRT else SMOKE_MODEL_PT

# The engine is built with a FIXED input shape. Inference must use the same
# size or TensorRT will reject the binding. Changing this requires re-exporting
# the engines with the matching imgsz.
IMGSZ = 640

# ---------------- Classes ----------------

FIRE_CLASS = "fire"
SMOKE_CLASS = "smoke"

# ---------------- Confidence ----------------

FIRE_CONF = 0.30
SMOKE_CONF = 0.55

# ---------------- Detection ----------------

# Inference passes per second, per camera.
#
# 30 is what pinned the GPU at 98% on a single camera. Fire and smoke develop
# over seconds, not milliseconds, so 5 fps costs nothing operationally (200 ms
# worst-case detection latency) and cuts GPU load roughly 6x. That is the
# difference between one camera and several on this board.
#
# Raise cautiously and re-check `tegrastats` — GR3D_FREQ should stay well
# under 70% so there is headroom for stream decode and the MJPEG encode.
DETECT_FPS = 5

ALARM_HOLD = 2

LOG_CLASSES = {"fire", "smoke"}

# --- Logging ---
CSV_FILE = "fire_smoke_logs.csv"
MAX_ALERTS = 50              # recent alerts kept in memory for the dashboard

# --- Web UI ---
WEB_PORT = 8000
JPEG_QUALITY = 25            # stream quality sent to browsers (1-100)

# --- IP cameras (auto-connected at startup) ---
# Credentials come from the environment so they never land in source control
# or in the startup log. Put them in a .env file (gitignored, chmod 600):
#     CAM_USER=admin
#     CAM_PASS=your_password_here     # @ must be written as %40, NO quotes
CAM_USER = os.environ.get("CAM_USER", "admin")
CAM_PASS = os.environ.get("CAM_PASS", "")

if not CAM_PASS:
    # Fail loudly rather than emitting rtsp://admin:@host and leaving you to
    # decode a wall of 401 Unauthorized messages.
    print("[config] WARNING: CAM_PASS is empty — RTSP connections will 401. "
          "Check that .env exists next to config.py and is unquoted.")


def _rtsp(host, channel=1, subtype=1):
    """Build an RTSP URL with credentials injected from the environment."""
    return (f"rtsp://{CAM_USER}:{CAM_PASS}@{host}:554"
            f"/video/live?channel={channel}&subtype={subtype}")


def mask_url(url):
    """Hide the password before printing or templating a stream URL."""
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, tail = rest.split("@", 1)
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:****@{tail}"


IP_CAMERAS = [
    (_rtsp("192.168.88.8",  channel=1, subtype=1), "Camera 1"),
    (_rtsp("192.168.88.12", channel=1, subtype=1), "Camera 2"),
    (_rtsp("192.168.88.6",  channel=1, subtype=1), "Camera 3"),
]

# --- MQTT sensors (ESP32 / DS18B20 etc.) ---
# SENTRY joins the SAME broker your ESP32 publishes to and subscribes as a
# read-only client. This is a supplementary environmental signal, NOT a
# certified heat detector (see PROJECT_CONTEXT.md safety positioning).
MQTT_BROKER    = "192.168.88.13"   # match DEFAULT_MQTT_SERVER in the ESP32 sketch
MQTT_PORT      = 1883
MQTT_KEEPALIVE = 60
MQTT_CLIENT_ID = "sentry-dashboard"
MQTT_USERNAME  = os.environ.get("MQTT_USERNAME") or None   # set if broker requires auth
MQTT_PASSWORD  = os.environ.get("MQTT_PASSWORD") or None

# How long (seconds) without a message before a sensor is shown as OFFLINE.
# The ESP32 publishes every ~3s, so 15s tolerates a few missed messages.
SENSOR_STALE_TIMEOUT = 15

# If True, a sensor at/above its alarm threshold also trips the dashboard's
# system-alarm banner. Default False = display only.
SENSOR_TRIPS_SYSTEM_ALARM = False

# One entry per MQTT topic. Add more ESP32 devices by adding rows.
#   id    : short unique key
#   name  : label shown in the UI
#   zone  : optional location text
#   topic : MQTT topic the device publishes to
#   unit  : shown after the value
#   warn  : value at/above which the tile turns amber (None = never)
#   alarm : value at/above which the tile turns red   (None = never)
MQTT_SENSORS = [
    {
        "id":    "BallPin",
        "name":  "BallPin Temp",
        "zone":  "Area 2",
        "topic": "balaji/xiao/temperature",
        "unit":  "°C",
        "warn":  45.0,
        "alarm": 60.0,
    },
    # {"id": "floor1_temp", "name": "Floor-1 Temp", "zone": "Line A",
    #  "topic": "balaji/floor1/temperature", "unit": "°C", "warn": 45, "alarm": 60},
]