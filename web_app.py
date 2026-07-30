"""
web_app.py  —  THE UI / SERVER LAYER
====================================
Flask routes only. Holds NO detection logic — it reads everything from a
Detector instance passed in via create_app(detector).

Public routes:
    /                 -> the SENTRY landing page (static product page)
    /login            -> sign-in page (GET form / POST credentials)
    /logout           -> clear the session

Protected routes (require login):
    /dashboard        -> the live operator console (templates/dashboard.html)
    /feed/<id>        -> live annotated MJPEG stream for one camera
    /api/status       -> JSON the page polls once a second
    /report           -> filtered PDF report download

Auth is local (auth.py): users + hashed passwords in users.json on this machine.
To change how the system LOOKS, edit the templates. To change WHAT IT DOES,
edit detector.py.
"""

import os
import time
from datetime import datetime
import cv2
from flask import (Flask, Response, jsonify, render_template, request,
                   send_from_directory, redirect, url_for, session)

import config
import report
import auth
import main_pc_sources

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_app(detector):
    app = Flask(__name__)
    app.secret_key = auth.get_secret_key()            # signs the login session cookie
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    # ---------- auth ----------
    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            u = request.form.get("username", "")
            p = request.form.get("password", "")
            nxt = request.form.get("next") or "/dashboard"
            if auth.verify(u, p):
                session["user"] = u
                return jsonify({"ok": True, "next": nxt})
            return jsonify({"ok": False, "error": "Incorrect username or password."}), 401
        # GET: send to the landing page with the login popup open
        nxt = request.args.get("next") or "/dashboard"
        return redirect("/?login=1&next=" + nxt)

    @app.route("/api/me")
    def whoami():
        # public: lets the landing page decide whether to pop the login modal
        return jsonify({"authenticated": bool(session.get("user")), "user": session.get("user")})

    @app.route("/logout")
    def logout():
        session.pop("user", None)
        return redirect("/")

    # ---------- public landing ----------
    @app.route("/")
    def landing():
        return send_from_directory(BASE_DIR, "landing.html")

    # ---------- live console (protected) ----------
    @app.route("/dashboard")
    @auth.login_required
    def index():
        return render_template("dashboard.html", cameras=detector.cameras, user=session.get("user"))

    def mjpeg(cam_id):
        # Send each annotated frame exactly once, the instant it's ready.
        # We track the detector's per-camera `seq`: if it hasn't advanced there
        # is no new frame, so we wait briefly instead of re-encoding/re-sending a
        # stale duplicate. This keeps the browser on the freshest frame and cuts
        # the added latency the old fixed-40ms loop introduced.
        last_seq = -1
        while not detector.stop_event.is_set():
            seq, frame = detector.get_latest(cam_id)
            if frame is None or seq == last_seq:
                time.sleep(0.005)      # no new frame yet — poll again shortly
                continue
            last_seq = seq
            ok, buf = cv2.imencode(".jpg", frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")

    @app.route("/feed/<int:cam_id>")
    @auth.login_required
    def feed(cam_id):
        if cam_id < 0 or cam_id >= len(detector.cameras):
            return "No camera", 404
        return Response(mjpeg(cam_id),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/status")
    @auth.login_required
    def api_status():
        return jsonify(detector.get_status())

    # ---------- camera management from the dashboard ----------
    @app.route("/api/discover", methods=["POST"])
    @auth.login_required
    def discover():
        # Query every known laptop, return cameras not already connected.
        try:
            found = main_pc_sources.discover_all()      # [(url, name), ...]
        except Exception as e:
            return jsonify({"ok": False, "error": str(e), "cameras": []}), 200
        existing = {c["url"] for c in detector.cameras}
        avail = [{"url": u, "name": n} for (u, n) in found if u not in existing]
        return jsonify({"ok": True, "cameras": avail})

    @app.route("/api/add_camera", methods=["POST"])
    @auth.login_required
    def add_camera_route():
        data = request.get_json(silent=True) or request.form
        url = (data.get("url") or "").strip()
        name = (data.get("name") or "").strip() or url
        if not url:
            return jsonify({"ok": False, "error": "Stream URL is required."}), 400
        res = detector.add_camera(url, name)
        if res is None:
            return jsonify({"ok": False, "error": "That camera is already connected."}), 409
        return jsonify({"ok": True, "camera": res})

    @app.route("/report")
    @auth.login_required
    def report_pdf():
        def parse_dt(val):
            if not val:
                return None
            for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d"):
                try:
                    return datetime.strptime(val, fmt)
                except ValueError:
                    continue
            return None

        start = parse_dt(request.args.get("start"))
        end = parse_dt(request.args.get("end"))
        camera = request.args.get("camera") or None

        pdf = report.build_report(start=start, end=end, camera=camera)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        fname = f"fire_report_{stamp}.pdf"
        return Response(pdf.read(), mimetype="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{fname}"'})

    return app