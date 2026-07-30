"""
config.py
=========
Shared settings for the whole system. Edit values here; both the detector
(algorithm) and the web UI read from this one place.
"""

# --- Model / detection ---
# ---------------- Models ----------------

FIRE_MODEL = "firebest.pt"
SMOKE_MODEL = "best.pt"

# ---------------- Classes ----------------

FIRE_CLASS = "fire"
SMOKE_CLASS = "smoke"

# ---------------- Confidence ----------------

FIRE_CONF = 0.30
SMOKE_CONF = 0.55

# ---------------- Detection ----------------

DETECT_FPS = 30
ALARM_HOLD = 2

LOG_CLASSES = {"fire", "smoke"}

# --- Logging ---
CSV_FILE = "fire_smoke_logs.csv"
MAX_ALERTS   = 50            # recent alerts kept in memory for the dashboard

# --- Web UI ---
WEB_PORT     = 8000
JPEG_QUALITY = 50            # stream quality sent to browsers (1-100)
# --- IP cameras (auto-connected at startup) ---
# These are added automatically when run.py starts.
# Format: (rtsp_url, display_name)
IP_CAMERAS = [
    ("rtsp://admin:Skay95111%40@192.168.88.8:554/video/live?channel=1&subtype=1",  "Camera 1"),
    ("rtsp://admin:Skay95111%40@192.168.88.12:554/video/live?channel=1&subtype=1", "Camera 2"),
    ("rtsp://admin:Skay95111%40@192.168.88.6:554/video/live?channel=1&subtype=1",  "Camera 3"),
    # add more cameras here...
]