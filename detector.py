"""
detector.py  —  THE ALGORITHM LAYER
===================================
All detection logic lives here. No web/UI code.

The Detector owns the models, one worker thread per camera, the latest
annotated frames, per-camera detection state, and the recent-alerts list.

Cameras can be added at runtime (from the dashboard). An optional MQTT sensor
hub (see sensors.py) can be injected; the detector only asks it for a snapshot
inside get_status(), so the UI seam stays a single call.

The UI layer reads from it via:
    det.cameras             -> list of camera dicts
    det.add_camera(url,name)-> add + start one camera at runtime
    det.has_camera(id)      -> bool
    det.get_frame(id)       -> latest annotated frame (numpy) or None
    det.get_latest(id)      -> (seq, frame) for duplicate-free MJPEG streaming
    det.get_status()        -> dict: cameras + sensors + alerts + system alarm
    det.start() / det.stop()

This file has zero knowledge of Flask, HTML, or MQTT plumbing.

--------------------------------------------------------------------------
TENSORRT NOTES
--------------------------------------------------------------------------
* Do NOT call .to("cuda") on a TensorRT model. The engine is already bound to
  the device it was compiled for; Ultralytics' AutoBackend handles placement.
  Calling .to() either errors or silently does nothing, and it misleads anyone
  reading the code into thinking device placement is managed here.

* Engines are loaded with an explicit task=. Unlike a .pt checkpoint, a
  compiled engine carries no task metadata, so Ultralytics cannot infer it.

* imgsz MUST match the shape the engine was built with. TensorRT bindings are
  fixed-shape; a mismatch is a hard failure, not a resize.

* Each engine holds ONE execution context, so concurrent calls from multiple
  camera threads must be serialised. That is what the locks are for — they are
  not optional here, and they mean added cameras contend for the shared
  engines rather than scaling linearly.

--------------------------------------------------------------------------
SMOKE DETECTION IS OPTIONAL
--------------------------------------------------------------------------
config.ENABLE_SMOKE gates the second model entirely: when False the engine is
never loaded, so it costs no device memory and no inference time. Everything
downstream handles smoke_model being None.

--------------------------------------------------------------------------
SENSOR HUB IS OPTIONAL
--------------------------------------------------------------------------
Pass a sensor hub into the constructor to surface MQTT sensor readings in
get_status():

    from sensors import SensorHub
    hub = SensorHub(...); hub.start()
    det = Detector(sensor_hub=hub)

The hub must expose snapshot() -> (readings_list, alarm_bool). When no hub is
given, get_status() still returns the `sensors` and `sensor_alarm` keys (empty
and False), so the dashboard never has to branch on their absence.

Set config.SENSOR_TRIPS_SYSTEM_ALARM = True to let a sensor alarm raise the
system-wide alarm on its own, independent of any camera detection.
"""

import cv2
import os
import csv
import time
import warnings
import threading
from datetime import datetime

# Ultralytics imports torch for pre/post-processing even on the TensorRT path,
# so this warning still prints on Jetson despite inference being native sm_87.
warnings.filterwarnings("ignore", message=".*compute capability.*")

import numpy as np
from ultralytics import YOLO, RTDETR
from camera_helper import CameraStream
import config


# Box colours per class.
COLOR = {
    "fire": (0, 0, 255),       # red
    "smoke": (0, 165, 255),    # amber
}


def _load_model(path, kind):
    """
    Load a .engine or .pt model.

    `kind` selects the wrapper class, which matters: RT-DETR uses a different
    predictor (no NMS, different output decoding) than YOLO. Loading an RT-DETR
    engine through the YOLO wrapper would apply the wrong post-processing and
    produce silently wrong boxes rather than an error.
    """
    is_engine = str(path).endswith(".engine")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            f"If this is an .engine file, build it on THIS device with:\n"
            f"  YOLO_AUTOINSTALL=false yolo export model=<weights>.pt "
            f"format=engine half=True device=0"
        )

    cls = RTDETR if kind == "rtdetr" else YOLO

    if is_engine:
        try:
            model = cls(path, task="detect")
        except (TypeError, AssertionError):
            # Some Ultralytics versions reject task= or the .engine suffix on
            # the RTDETR wrapper. Fall back to YOLO, which routes through the
            # same AutoBackend. Verify your smoke boxes look sane if you hit
            # this path — the post-processing may not match RT-DETR's output.
            print(f"[detector] WARNING: {cls.__name__} rejected {path}, "
                  f"falling back to YOLO wrapper. Verify detections.")
            model = YOLO(path, task="detect")
    else:
        model = cls(path)
        model.to("cuda")        # only meaningful for the PyTorch path

    print(f"[detector] loaded {path} ({'TensorRT' if is_engine else 'PyTorch'})")
    return model


class Detector:
    def __init__(self, sensor_hub=None):
        print("Loading models...")
        self.fire_model = _load_model(config.FIRE_MODEL, "yolo")

        if config.ENABLE_SMOKE:
            self.smoke_model = _load_model(config.SMOKE_MODEL, "rtdetr")
        else:
            self.smoke_model = None
            print("[detector] smoke detection DISABLED (config.ENABLE_SMOKE)")

        # Serialise access: one TensorRT execution context per engine.
        self.fire_lock = threading.Lock()
        self.smoke_lock = threading.Lock()

        self._warmup()

        self.cameras = []          # [{id, name, url, stream, annotated, last_det, lock}]
        self.cameras_lock = threading.Lock()   # guards add + the cameras list
        self._next_id = 0

        self.alerts = []           # recent detections (newest first)
        self.sensor_hub = sensor_hub   # optional MQTT sensor source (or None)
        self.state_lock = threading.Lock()
        self.csv_lock = threading.Lock()
        self.stop_event = threading.Event()
        self._threads = []

        if self.sensor_hub is not None:
            print("[detector] sensor hub attached")

        self._init_csv()

    # ---------- setup ----------
    def _warmup(self):
        """
        Run one pass through each loaded engine so the first live frame isn't
        delayed by execution-context allocation.
        """
        blank = np.zeros((config.IMGSZ, config.IMGSZ, 3), dtype=np.uint8)
        t0 = time.time()
        with self.fire_lock:
            self.fire_model(blank, imgsz=config.IMGSZ, verbose=False)
        if self.smoke_model is not None:
            with self.smoke_lock:
                self.smoke_model(blank, imgsz=config.IMGSZ, verbose=False)
        print(f"[detector] warm-up complete in {time.time() - t0:.1f}s")

    def _init_csv(self):
        if not os.path.exists(config.CSV_FILE):
            with open(config.CSV_FILE, "w", newline="") as f:
                csv.writer(f).writerow(["Date", "Time", "Camera", "Detection", "Confidence"])

    # ---------- camera management ----------
    def add_camera(self, url, name=None):
        """Add ONE camera and start its worker immediately. Returns {id,name} or
        None if that URL is already connected."""
        with self.cameras_lock:
            if any(c["url"] == url for c in self.cameras):
                return None
            cid = self._next_id
            self._next_id += 1
            cam = {"id": cid, "name": name or config.mask_url(url), "url": url,
                   "stream": None, "annotated": None, "seq": 0, "last_det": None,
                   "lock": threading.Lock()}
            self.cameras.append(cam)

        t = threading.Thread(target=self._worker, args=(cam,), daemon=True)
        t.start()
        self._threads.append(t)
        # Mask the credential before it reaches the log.
        print(f"[+] camera added: {cam['name']} -> {config.mask_url(url)}")
        return {"id": cid, "name": cam["name"]}

    def add_sources(self, sources):
        """sources: [(url, name), ...]  — bulk add (used at startup)."""
        for url, name in sources:
            self.add_camera(url, name)

    def has_camera(self, cam_id):
        return self._cam_by_id(cam_id) is not None

    def _cam_by_id(self, cam_id):
        for c in self.cameras:
            if c["id"] == cam_id:
                return c
        return None

    # ---------- logging ----------
    def _log_csv(self, name, det, conf):
        now = datetime.now()
        with self.csv_lock:
            with open(config.CSV_FILE, "a", newline="") as f:
                csv.writer(f).writerow([now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
                                        name, det, f"{conf:.2f}"])

    def _add_alert(self, name, det, conf):
        entry = {"time": datetime.now().strftime("%H:%M:%S"), "camera": name,
                 "type": det, "conf": round(conf, 2)}
        with self.state_lock:
            self.alerts.insert(0, entry)
            del self.alerts[config.MAX_ALERTS:]
        self._log_csv(name, det, conf)

    # ---------- annotation ----------
    def _draw_box(self, annotated, x1, y1, x2, y2, cname, conf):
        """Draw one labelled detection box with a filled label background."""
        color = COLOR.get(cname, (0, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{cname.upper()}  {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        # Clamp to the frame so the label doesn't vanish for boxes at the top.
        top = max(y1, th + baseline + 6)
        cv2.rectangle(annotated, (x1, top - th - baseline - 6),
                      (x1 + tw + 6, top), color, -1)
        cv2.putText(annotated, label, (x1 + 3, top - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # ---------- per-camera worker ----------
    def _worker(self, cam):
        stream = CameraStream(cam["url"], name=cam["name"]).start()
        cam["stream"] = stream
        last_detect = 0.0
        last_log = {}
        prev_time = time.time()

        while not self.stop_event.is_set():
            ok, frame = stream.read()
            if not ok:
                time.sleep(0.03)
                continue

            # Read DETECT_FPS every loop so a live /config page can change it
            # without a restart.
            fps_target = config.DETECT_FPS
            period = 1.0 / fps_target if fps_target > 0 else 0

            now = time.time()
            if (now - last_detect) < period:
                time.sleep(0.005)
                continue
            last_detect = now

            # imgsz is explicit: the engine's input binding is fixed-shape and
            # will reject anything else.
            try:
                with self.fire_lock:
                    fire_results = self.fire_model(
                        frame, conf=config.FIRE_CONF,
                        imgsz=config.IMGSZ, verbose=False,
                    )

                smoke_results = None
                if self.smoke_model is not None:
                    with self.smoke_lock:
                        smoke_results = self.smoke_model(
                            frame, conf=config.SMOKE_CONF,
                            imgsz=config.IMGSZ, verbose=False,
                        )
            except Exception as e:
                # One bad frame should not kill the worker and take the camera
                # offline for the rest of the run.
                print(f"[{cam['name']}] inference error: {e}")
                time.sleep(0.1)
                continue

            annotated = frame.copy()

            # ---------------- FIRE ----------------
            for box in fire_results[0].boxes:
                cls = int(box.cls[0])
                cname = self.fire_model.names[cls].lower()
                if cname != config.FIRE_CLASS:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cam["last_det"] = {"type": "fire", "conf": conf, "t": now}
                if now - last_log.get("fire", 0) >= 1.0:
                    self._add_alert(cam["name"], "fire", conf)
                    last_log["fire"] = now

                self._draw_box(annotated, x1, y1, x2, y2, "fire", conf)

            # ---------------- SMOKE (skipped when disabled) ----------------
            if smoke_results is not None:
                for box in smoke_results[0].boxes:
                    cls = int(box.cls[0])
                    cname = self.smoke_model.names[cls].lower()
                    if cname != config.SMOKE_CLASS:
                        continue

                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cam["last_det"] = {"type": "smoke", "conf": conf, "t": now}
                    if now - last_log.get("smoke", 0) >= 1.0:
                        self._add_alert(cam["name"], "smoke", conf)
                        last_log["smoke"] = now

                    self._draw_box(annotated, x1, y1, x2, y2, "smoke", conf)

            # ---------------- FPS overlay ----------------
            # This is the DETECTION rate for this camera, not the stream's frame
            # rate. It should read close to DETECT_FPS; if it falls well below
            # as cameras are added, they are contending for the shared engine
            # and it is time to check tegrastats.
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if current_time > prev_time else 0.0
            prev_time = current_time
            cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            with cam["lock"]:
                cam["annotated"] = annotated
                cam["seq"] += 1        # marks this frame as fresh for streamers

        stream.stop()

    # ---------- lifecycle ----------
    def start(self):
        # Cameras start when added, so this is kept only for API compatibility.
        pass

    def stop(self):
        self.stop_event.set()
        # Give workers a moment to notice the flag and release their captures
        # before the interpreter tears down around them.
        for t in self._threads:
            t.join(timeout=3.0)

    # ---------- read API (used by the UI layer) ----------
    def get_frame(self, cam_id):
        cam = self._cam_by_id(cam_id)
        if cam is None:
            return None
        with cam["lock"]:
            return None if cam["annotated"] is None else cam["annotated"].copy()

    def get_latest(self, cam_id):
        """Return (seq, frame_copy) for the newest annotated frame, or (None, None).
        `seq` lets a streamer send each frame exactly once and skip duplicates,
        so the browser always gets the freshest frame with minimal added lag."""
        cam = self._cam_by_id(cam_id)
        if cam is None:
            return None, None
        with cam["lock"]:
            if cam["annotated"] is None:
                return None, None
            return cam["seq"], cam["annotated"].copy()

    def get_status(self):
        now = time.time()
        cams = []
        system_alarm = False
        alarm_cam = None
        for c in list(self.cameras):          # snapshot: safe while adding
            ld = c.get("last_det")
            in_alarm = bool(ld and (now - ld["t"]) <= config.ALARM_HOLD)
            online = c["stream"].healthy() if c.get("stream") else False
            if in_alarm:
                system_alarm = True
                alarm_cam = c["name"]
            cams.append({
                "id": c["id"], "name": c["name"], "online": online,
                "alarm": in_alarm,
                "type": ld["type"] if in_alarm else None,
                "conf": round(ld["conf"], 2) if in_alarm else None,
            })

        # --- MQTT sensors (optional) ---
        # The keys are always present so the dashboard never has to branch on
        # whether a hub was configured. A hub that throws must not take down
        # the whole status endpoint — the cameras are the critical path.
        sensors, sensor_alarm = ([], False)
        if self.sensor_hub is not None:
            try:
                sensors, sensor_alarm = self.sensor_hub.snapshot()
            except Exception as e:
                print(f"[detector] sensor hub snapshot failed: {e}")
                sensors, sensor_alarm = ([], False)

            if getattr(config, "SENSOR_TRIPS_SYSTEM_ALARM", False) and sensor_alarm:
                system_alarm = True
                alarm_cam = alarm_cam or "OVER-TEMP"

        with self.state_lock:
            recent = list(self.alerts)

        return {
            "cameras": cams,
            "alerts": recent,
            "sensors": sensors,
            "sensor_alarm": sensor_alarm,
            "system_alarm": system_alarm,
            "alarm_cam": alarm_cam,
            "smoke_enabled": config.ENABLE_SMOKE,   # so the UI can hide smoke widgets
            "online": sum(1 for c in cams if c["online"]),
            "total": len(cams),
        }