"""
run.py  —  ENTRY POINT
======================
Applies saved settings -> start MQTT sensor hub -> build the Detector
(algorithm) -> auto-connect cameras -> hand it to the web app (UI) -> serve.

Run order:
    1. Start laptop_camera_server.py on each office laptop (optional).
    2. python run.py  -> open the printed URL.

Layers it connects:
    settings_store.py = live-editable config overlay (settings.json)
    sensors.py    = MQTT sensor source (optional)
    detector.py   = algorithm
    web_app.py    = server      } UI
    templates/…   = look        }
"""

import socket

from detector import Detector
from web_app import create_app
import config
import settings_store
import auth
import main_pc_sources


def get_lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def start_sensor_hub():
    """Bring up MQTT sensors if paho-mqtt is installed; otherwise skip cleanly."""
    try:
        from sensors import SensorHub
    except ImportError as e:
        print(f"[run] MQTT sensors disabled (paho-mqtt not installed): {e}")
        print("[run] install with:  pip install paho-mqtt")
        return None
    return SensorHub().start()


def main():
    # Apply any saved overrides from settings.json BEFORE anything reads config.
    # (model path, broker, port are read at construction time below.)
    settings_store.load_and_apply()

    sensor_hub = start_sensor_hub()

    detector = Detector(sensor_hub=sensor_hub)
    for url, name in config.IP_CAMERAS:
        detector.add_camera(url, name)

    # Best-effort laptop auto-connect. Never fatal.
    try:
        sources = main_pc_sources.discover_all()
    except Exception as e:
        print(f"Startup discovery skipped: {e}")
        sources = []
    if sources:
        detector.add_sources(sources)
    detector.start()

    app = create_app(detector)

    seeded = auth.ensure_default()

    ip = get_lan_ip()
    print("\n===================================")
    print(" SENTRY RUNNING")
    print("===================================")
    print(f" Landing page :  http://{ip}:{config.WEB_PORT}/")
    print(f" Live console :  http://{ip}:{config.WEB_PORT}/dashboard")
    print(f" Settings     :  http://{ip}:{config.WEB_PORT}/config")
    print(f" Cameras connected at start: {len(detector.cameras)}")
    if sensor_hub is not None:
        print(f" Sensors      :  {len(sensor_hub.sensors)} "
              f"(MQTT {config.MQTT_BROKER}:{config.MQTT_PORT})")
    if not detector.cameras:
        print(" (none found — add them from the dashboard 'Add cameras' button)")
    if seeded:
        print("-----------------------------------")
        print(" FIRST RUN — default login created:")
        print("   username: admin")
        print("   password: changeme")
        print(" CHANGE IT NOW:  python auth.py add admin")
    print("===================================\n")

    try:
        app.run(host="0.0.0.0", port=config.WEB_PORT, threaded=True)
    finally:
        detector.stop()
        if sensor_hub is not None:
            sensor_hub.stop()


if __name__ == "__main__":
    main()