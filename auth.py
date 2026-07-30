"""
auth.py  —  LOCAL LOGIN / AUTH LAYER
====================================
Username + password auth, stored entirely on this machine. No cloud, no
external service. Passwords are HASHED (never stored in plaintext) using
Werkzeug, which ships with Flask.

Files it manages (created automatically next to this file):
    users.json   -> { "username": "<password-hash>", ... }
    secret.key   -> random key used to sign session cookies (persists logins
                    across restarts)

Manage users from the command line:
    python auth.py add <username>      # create or reset a user (prompts for password)
    python auth.py list                # list usernames
    python auth.py remove <username>   # delete a user

The web layer uses:
    auth.get_secret_key()        -> str   (set as app.secret_key)
    auth.login_required          -> route decorator
    auth.verify(user, password)  -> bool
    auth.ensure_default()        -> bool  (seeds admin/changeme if no users yet)
"""

import os
import json
import functools
import secrets

from flask import session, redirect, url_for, request
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")
SECRET_FILE = os.path.join(BASE_DIR, "secret.key")


# ---------- secret key (signs session cookies) ----------
def get_secret_key():
    if os.path.exists(SECRET_FILE):
        with open(SECRET_FILE) as f:
            return f.read().strip()
    key = secrets.token_hex(32)
    with open(SECRET_FILE, "w") as f:
        f.write(key)
    return key


# ---------- user store ----------
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE) as f:
            return json.load(f)
    except (ValueError, OSError):
        return {}


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def add_user(username, password):
    users = load_users()
    users[username] = generate_password_hash(password)
    save_users(users)


def remove_user(username):
    users = load_users()
    if username in users:
        del users[username]
        save_users(users)
        return True
    return False


def verify(username, password):
    users = load_users()
    h = users.get(username)
    return bool(h and check_password_hash(h, password))


def ensure_default():
    """Seed a default admin if no users exist yet. Returns True if it created one."""
    if not load_users():
        add_user("admin", "changeme")
        return True
    return False


# ---------- web helpers ----------
def current_user():
    return session.get("user")


def login_required(view):
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            p = request.path
            # background requests (data/stream) get a clean 401, not an HTML page
            if p.startswith("/api/") or p.startswith("/feed/"):
                return ("Unauthorized", 401)
            # pages bounce to the landing page with the login popup open
            return redirect("/?login=1&next=" + p)
        return view(*args, **kwargs)
    return wrapped


# ---------- command-line user management ----------
if __name__ == "__main__":
    import sys
    import getpass

    args = sys.argv[1:]
    cmd = args[0] if args else ""

    if cmd == "add" and len(args) >= 2:
        user = args[1]
        pw1 = getpass.getpass(f"Password for '{user}': ")
        pw2 = getpass.getpass("Confirm password: ")
        if pw1 != pw2:
            print("Passwords do not match.")
            sys.exit(1)
        if not pw1:
            print("Password cannot be empty.")
            sys.exit(1)
        add_user(user, pw1)
        print(f"User '{user}' saved.")

    elif cmd == "list":
        users = load_users()
        if not users:
            print("No users yet. Add one with: python auth.py add <username>")
        else:
            print("Users:")
            for u in sorted(users):
                print(f"  - {u}")

    elif cmd == "remove" and len(args) >= 2:
        if remove_user(args[1]):
            print(f"Removed '{args[1]}'.")
        else:
            print(f"No such user: {args[1]}")

    else:
        print("Usage:")
        print("  python auth.py add <username>      create or reset a user")
        print("  python auth.py list                list usernames")
        print("  python auth.py remove <username>   delete a user")