"""
sensors.py  —  THE MQTT SENSOR LAYER
====================================
All MQTT plumbing lives here. No web/UI code, no detection logic.

Mirrors camera_helper.py: that reads camera streams, this reads sensor topics.
An ESP32 (DS18B20, etc.) publishes readings to an MQTT broker; SensorHub
subscribes and keeps the latest value per topic, with freshness tracking.

Touch it through only:
    hub = SensorHub().start()
    readings, any_alarm = hub.snapshot()            # Detector.get_status()
    hub.update_thresholds(id, warn, alarm)          # live edits from /config
    hub.stop()

Works with paho-mqtt 2.x (current) and 1.x. Install: pip install paho-mqtt
Set MQTT_DEBUG = True in config.py for paho's internal packet logs.
"""

import time
import threading

import paho.mqtt.client as mqtt
import config


def _make_client(client_id):
    if hasattr(mqtt, "CallbackAPIVersion"):          # paho-mqtt >= 2.0
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                           client_id=client_id, clean_session=True)
    return mqtt.Client(client_id=client_id, clean_session=True)   # paho 1.x


class SensorHub:
    def __init__(self):
        self._lock = threading.Lock()
        self.sensors = {}        # id -> live state dict
        self._topic_index = {}   # topic -> sensor id

        for s in getattr(config, "MQTT_SENSORS", []):
            self.sensors[s["id"]] = {
                "id":    s["id"],
                "name":  s["name"],
                "zone":  s.get("zone", ""),
                "topic": s["topic"],
                "unit":  s.get("unit", ""),
                "warn":  s.get("warn"),
                "alarm": s.get("alarm"),
                "value": None,
                "raw":   None,
                "updated": 0.0,
            }
            self._topic_index[s["topic"]] = s["id"]

        self._client = _make_client(config.MQTT_CLIENT_ID)
        if getattr(config, "MQTT_USERNAME", None):
            self._client.username_pw_set(config.MQTT_USERNAME,
                                         getattr(config, "MQTT_PASSWORD", None))
        if getattr(config, "MQTT_DEBUG", False):
            self._client.enable_logger()
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)

    # ---------- lifecycle ----------
    def start(self):
        if not self.sensors:
            print("[sensors] no MQTT_SENSORS configured; hub idle.")
            return self

        target = f"{config.MQTT_BROKER}:{config.MQTT_PORT}"
        try:
            self._client.connect(config.MQTT_BROKER, config.MQTT_PORT,
                                 config.MQTT_KEEPALIVE)
            print(f"[sensors] MQTT TCP connect to {target} OK "
                  f"({len(self.sensors)} sensor(s))")
        except Exception as e:
            print(f"[sensors] *** MQTT connect to {target} FAILED: "
                  f"{type(e).__name__}: {e}")
            print(f"[sensors] *** check the broker IP is reachable FROM THIS PC. "
                  f"Will keep retrying in the background.")
            try:
                self._client.connect_async(config.MQTT_BROKER, config.MQTT_PORT,
                                           config.MQTT_KEEPALIVE)
            except Exception as e2:
                print(f"[sensors] connect_async also failed: {e2}")

        self._client.loop_start()
        return self

    def stop(self):
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass

    # ---------- live threshold edits (from /config) ----------
    def update_thresholds(self, sensor_id, warn, alarm):
        with self._lock:
            s = self.sensors.get(sensor_id)
            if s is not None:
                s["warn"] = warn
                s["alarm"] = alarm

    # ---------- callbacks (cover paho 1.x AND 2.x signatures) ----------
    def _on_connect(self, client, userdata, flags, reason_code, properties=None, *a):
        rc = getattr(reason_code, "value", reason_code)
        if rc == 0:
            for s in self.sensors.values():
                client.subscribe(s["topic"])
            print(f"[sensors] MQTT connected; subscribed to "
                  f"{[s['topic'] for s in self.sensors.values()]}")
        else:
            print(f"[sensors] *** MQTT connect REFUSED (code {rc}). "
                  f"Code 4/5 usually means a username/password is required.")

    def _on_disconnect(self, client, userdata, *a):
        print(f"[sensors] MQTT disconnected {a} — auto-reconnecting.")

    def _on_message(self, client, userdata, msg):
        sid = self._topic_index.get(msg.topic)
        if sid is None:
            return
        raw = msg.payload.decode(errors="replace").strip()
        try:
            value = float(raw)
        except ValueError:
            value = None
        with self._lock:
            s = self.sensors[sid]
            s["raw"] = raw
            s["value"] = value
            s["updated"] = time.time()

    # ---------- read API ----------
    def snapshot(self):
        now = time.time()
        out = []
        any_alarm = False
        stale_after = getattr(config, "SENSOR_STALE_TIMEOUT", 15)

        for s in self.sensors.values():
            with self._lock:
                value, raw, updated = s["value"], s["raw"], s["updated"]
                warn, alarm = s["warn"], s["alarm"]

            fresh = updated > 0 and (now - updated) <= stale_after

            if not fresh:
                status = "offline"
            elif value is not None and alarm is not None and value >= alarm:
                status = "alarm"
                any_alarm = True
            elif value is not None and warn is not None and value >= warn:
                status = "warn"
            else:
                status = "ok"

            out.append({
                "id": s["id"], "name": s["name"], "zone": s["zone"],
                "unit": s["unit"], "value": value, "raw": raw,
                "status": status, "fresh": fresh,
                "age": round(now - updated, 1) if updated else None,
            })
        return out, any_alarm