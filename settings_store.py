"""
settings_store.py  —  LIVE-EDITABLE SETTINGS OVERLAY
====================================================
Lets a small, safe subset of config.py be edited from the /config web page
instead of in code. Edited values are:

  * applied onto the `config` module at runtime (per-call reads pick them up)
  * persisted to settings.json (so they survive a restart)

This module NEVER rewrites config.py. It only overlays a JSON file, which is
far safer than editing/executing Python from a web form. Values not listed in
SCALAR_SPEC stay code-only in config.py.

At startup, run.py calls load_and_apply() BEFORE building the Detector and the
SensorHub, so even "restart-required" values (model path, broker, port) take
effect on the next run.
"""

import os
import json
import threading

import config

SETTINGS_FILE = "settings.json"
_lock = threading.Lock()


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# key -> (caster, applies_live, clamp(lo,hi) | None, label, group)
#   applies_live=True  -> read per-call in the running system, edits are instant
#   applies_live=False -> read once at startup, edits need a restart
SCALAR_SPEC = {
    "CONF":                      (float, True,  (0.01, 1.0),   "Detection confidence",        "detection"),
    "DETECT_FPS":                (float, True,  (0.1, 120.0),  "Detection FPS / camera",      "detection"),
    "ALARM_HOLD":                (float, True,  (0.0, 3600.0), "Alarm hold (seconds)",        "detection"),
    "JPEG_QUALITY":              (int,   True,  (1, 100),      "Stream JPEG quality",         "stream"),
    "MAX_ALERTS":                (int,   True,  (1, 1000),     "Alerts kept in log",          "stream"),
    "SENSOR_STALE_TIMEOUT":      (float, True,  (1.0, 600.0),  "Sensor stale timeout (s)",    "sensors"),
    "SENSOR_TRIPS_SYSTEM_ALARM": (bool,  True,  None,          "Over-temp trips system alarm","sensors"),
    # --- restart required (read once at startup) ---
    "MODEL_PATH":  (str, False, None,         "YOLO model path",   "connection"),
    "WEB_PORT":    (int, False, (1, 65535),   "Web port",          "connection"),
    "MQTT_BROKER": (str, False, None,         "MQTT broker IP",    "connection"),
    "MQTT_PORT":   (int, False, (1, 65535),   "MQTT broker port",  "connection"),
}


def _num_or_none(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _cast(key, raw):
    caster, live, clamp, label, group = SCALAR_SPEC[key]
    if caster is bool:
        return raw if isinstance(raw, bool) else str(raw).strip().lower() in ("1", "true", "yes", "on")
    val = caster(raw)
    if clamp:
        val = _clamp(val, clamp[0], clamp[1])
    return val


# ---------- json io ----------
def _read_json():
    if not os.path.exists(SETTINGS_FILE):
        return {}
    try:
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    except Exception as e:
        print(f"[settings] could not read {SETTINGS_FILE}: {e}")
        return {}


def _write_json(data):
    tmp = SETTINGS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, SETTINGS_FILE)   # atomic-ish replace


# ---------- apply ----------
def _apply_scalars(scalars):
    applied = []
    for key, raw in scalars.items():
        if key not in SCALAR_SPEC:
            continue
        try:
            val = _cast(key, raw)
        except Exception:
            continue
        setattr(config, key, val)
        applied.append(key)
    return applied


def _apply_sensor_thresholds(thresholds, sensor_hub=None):
    """thresholds: {sensor_id: {"warn": x|None, "alarm": y|None}}"""
    by_id = {s["id"]: s for s in getattr(config, "MQTT_SENSORS", [])}
    for sid, t in thresholds.items():
        warn = _num_or_none(t.get("warn"))
        alarm = _num_or_none(t.get("alarm"))
        if sid in by_id:                 # source of truth for next startup
            by_id[sid]["warn"] = warn
            by_id[sid]["alarm"] = alarm
        if sensor_hub is not None and hasattr(sensor_hub, "update_thresholds"):
            sensor_hub.update_thresholds(sid, warn, alarm)   # live update


def load_and_apply(sensor_hub=None):
    """Called by run.py at startup, before Detector/SensorHub are built."""
    data = _read_json()
    if not data:
        return
    applied = _apply_scalars(data.get("scalars", {}))
    _apply_sensor_thresholds(data.get("sensor_thresholds", {}), sensor_hub)
    print(f"[settings] applied overrides from {SETTINGS_FILE}: {applied}")


# ---------- web API helpers ----------
def current_editable(detector=None):
    """Everything the /config page needs to render the form with live values."""
    scalars, meta = {}, {}
    for key, (caster, live, clamp, label, group) in SCALAR_SPEC.items():
        scalars[key] = getattr(config, key, None)
        if caster is bool:
            field_type = "bool"
        elif caster is int:
             field_type = "int"
        elif caster is float:
            field_type = "number"
        else:
            field_type = "text"

        meta[key] = {
    "live": live,
    "label": label,
    "group": group,
    "type": field_type,
    "min": clamp[0] if clamp else None,
    "max": clamp[1] if clamp else None,
}
    sensors = [{
        "id": s["id"], "name": s["name"], "unit": s.get("unit", ""),
        "warn": s.get("warn"), "alarm": s.get("alarm"),
    } for s in getattr(config, "MQTT_SENSORS", [])]
    return {"scalars": scalars, "meta": meta, "sensors": sensors}


def save_and_apply(data, detector=None):
    """Validate, apply live, persist. Returns a small result dict for the UI."""
    scalars = data.get("scalars", {}) or {}
    thresholds = data.get("sensor_thresholds", {}) or {}
    sensor_hub = getattr(detector, "sensor_hub", None) if detector else None

    with _lock:
        applied = _apply_scalars(scalars)
        _apply_sensor_thresholds(thresholds, sensor_hub)

        existing = _read_json()
        existing.setdefault("scalars", {}).update({k: getattr(config, k) for k in applied})
        st = existing.setdefault("sensor_thresholds", {})
        for sid, t in thresholds.items():
            st[sid] = {"warn": _num_or_none(t.get("warn")), "alarm": _num_or_none(t.get("alarm"))}
        _write_json(existing)

    restart_required = [k for k in applied if not SCALAR_SPEC[k][1]]
    return {"ok": True, "applied": applied, "restart_required": restart_required}