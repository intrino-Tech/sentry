"""
run.py  —  ENTRY POINT
======================
Wires everything together: pick laptops -> build the Detector (algorithm)
-> hand it to the web app (UI) -> serve.

Run order:
    1. Start laptop_camera_server.py on each office laptop.
    2. python run.py  -> choose laptops  -> open the printed URL.

The two layers it connects:
    detector.py   = algorithm  (edit to change detection behaviour)
    web_app.py    = server      } UI
    templates/dashboard.html    } (edit to change the look)
"""

import socket

from detector import Detector
from web_app import create_app
import config
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


def main():
    detector = Detector()
    for url, name in config.IP_CAMERAS:
        detector.add_camera(url, name)

    # Best-effort: try to auto-connect known laptops at startup. Never fatal —
    # the server starts even if none are found, and you can add cameras from
    # the dashboard's "Add cameras" button.
    try:
        sources = main_pc_sources.discover_all()
    except Exception as e:
        print(f"Startup discovery skipped: {e}")
        sources = []
    if sources:
        detector.add_sources(sources)
    detector.start()

    app = create_app(detector)

    # Seed a default login on first run so you're never locked out.
    seeded = auth.ensure_default()

    ip = get_lan_ip()
    print("\n===================================")
    print(" SENTRY RUNNING")
    print("===================================")
    print(f" Landing page :  http://{ip}:{config.WEB_PORT}/")
    print(f" Live console :  http://{ip}:{config.WEB_PORT}/dashboard")
    print(f" Cameras connected at start: {len(detector.cameras)}")
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


if __name__ == "__main__":
    main()