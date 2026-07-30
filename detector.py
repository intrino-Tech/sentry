"""
detector.py  —  THE ALGORITHM LAYER
===================================
All detection logic lives here. No web/UI code.

The Detector owns the YOLO model, one worker thread per camera, the latest
annotated frames, per-camera detection state, and the recent-alerts list.

Cameras can be added at runtime (from the dashboard) — add_camera() starts a
worker on the fly, so the system runs fine starting with zero cameras.

The UI layer reads from it via:
    det.cameras            -> list of camera dicts
    det.add_camera(url,name)-> add + start one camera at runtime
    det.has_camera(id)     -> bool
    det.get_frame(id)      -> latest annotated frame (numpy) or None
    det.get_status()       -> dict: per-camera status + alerts + system alarm
    det.start() / det.stop()

This file has zero knowledge of Flask, HTML, or how it's displayed.
"""
import cv2
import os
import csv
import time
import threading
from datetime import datetime

from ultralytics import YOLO, RTDETR
from camera_helper import CameraStream
import config


class Detector:
    def __init__(self):
        print("Loading model...")
        self.fire_model = YOLO(config.FIRE_MODEL)
        self.smoke_model = RTDETR(config.SMOKE_MODEL)

        self.fire_model.to("cuda")
        self.smoke_model.to("cuda")

        self.fire_lock = threading.Lock()
        self.smoke_lock = threading.Lock()
        

        self.cameras = []          # [{id, name, url, stream, annotated, last_det, lock}]
        self.cameras_lock = threading.Lock()   # guards add + the cameras list
        self._next_id = 0

        self.alerts = []           # recent detections (newest first)
        self.state_lock = threading.Lock()
        self.csv_lock = threading.Lock()
        self.stop_event = threading.Event()
        self._threads = []

        self._init_csv()

    # ---------- setup ----------
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
            cam = {"id": cid, "name": name or url, "url": url, "stream": None,
                   "annotated": None, "seq": 0, "last_det": None,
                   "lock": threading.Lock()}
            self.cameras.append(cam)

        t = threading.Thread(target=self._worker, args=(cam,), daemon=True)
        t.start()
        self._threads.append(t)
        print(f"[+] camera added: {cam['name']} -> {url}")
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

    # ---------- per-camera worker ----------
    def _worker(self, cam):
        stream = CameraStream(cam["url"], name=cam["name"]).start()
        cam["stream"] = stream
        period = 1.0 / config.DETECT_FPS if config.DETECT_FPS > 0 else 0
        last_detect = 0.0
        last_log = {}
        prev_time = time.time()

        while not self.stop_event.is_set():
            ok, frame = stream.read()
            if not ok:
                time.sleep(0.03)
                continue

            now = time.time()
            if (now - last_detect) >= period:
                last_detect = now
                with self.fire_lock:
                    fire_results = self.fire_model(
                        frame,
                        conf=config.FIRE_CONF,
                        verbose=False,
                        device=0
                    )

                with self.smoke_lock:
                    smoke_results = self.smoke_model(
                        frame,
                        conf=config.SMOKE_CONF,
                        verbose=False,
                        device=0
                    )
                annotated = frame.copy()

                # ---------------- FIRE ----------------
                for box in fire_results[0].boxes:

                    cls = int(box.cls[0])
                    cname = self.fire_model.names[cls].lower()

                    if cname != config.FIRE_CLASS:
                        continue

                    conf = float(box.conf[0])

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cam["last_det"] = {
                        "type": "fire",
                        "conf": conf,
                        "t": now
                    }

                    if now - last_log.get("fire", 0) >= 1.0:
                        self._add_alert(cam["name"], "fire", conf)
                        last_log["fire"] = now

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    cv2.putText(
                        annotated,
                        f"Fire {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )


                # ---------------- SMOKE ----------------
                for box in smoke_results[0].boxes:

                    cls = int(box.cls[0])
                    cname = self.smoke_model.names[cls].lower()

                    if cname != config.SMOKE_CLASS:
                        continue

                    conf = float(box.conf[0])

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    cam["last_det"] = {
                        "type": "smoke",
                        "conf": conf,
                        "t": now
                    }

                    if now - last_log.get("smoke", 0) >= 1.0:
                        self._add_alert(cam["name"], "smoke", conf)
                        last_log["smoke"] = now

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    cv2.putText(
                        annotated,
                        f"Smoke {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )

                # ---------------- FPS overlay ----------------
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if current_time > prev_time else 0.0
                prev_time = current_time
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

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
        with self.state_lock:
            recent = list(self.alerts)
        return {
            "cameras": cams,
            "alerts": recent,
            "system_alarm": system_alarm,
            "alarm_cam": alarm_cam,
            "online": sum(1 for c in cams if c["online"]),
            "total": len(cams),
        }