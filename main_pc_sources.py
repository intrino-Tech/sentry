"""
main_pc_sources.py
==================
RUNS ON THE MAIN DETECTION PC.

Lets you CHOOSE which office laptops to pull cameras from (any number), then
queries each chosen laptop's /list endpoint and collects ALL their cameras into
one flat list of (url, name) sources to feed your YOLO pipeline.

Workflow:
  1. Register your office laptops once in KNOWN_LAPTOPS below (name + IP).
  2. Run this file -> it shows the list -> you pick which ones to use.
  3. It returns every camera from the chosen laptops.
"""

import requests   # pip install requests

# ==========================
# Register your office laptops here (once). Add as many as you have.
# name is just a friendly label; ip is the laptop's LAN IP.
# ==========================
KNOWN_LAPTOPS = [
    # {"name": "Reception Laptop", "ip": "192.168.88.6"},
    # {"name": "Floor-1 Laptop",   "ip": "192.168.88.8"},
    # {"name": "Floor-2 Laptop",   "ip": "192.168.88.12"},
    # {"name": "Lab Laptop",       "ip": "192.168.1.45"},
    # # add more...
]

PORT = 5000
TIMEOUT = 3.0   # seconds to wait per laptop


# ==========================
# Interactive chooser
# ==========================
def choose_laptops():
    """Show the known laptops and let the user pick any number of them."""
    print("\n===================================")
    print(" Available Office Laptops")
    print("===================================")
    for i, lap in enumerate(KNOWN_LAPTOPS, start=1):
        print(f"  {i}. {lap['name']:<20} {lap['ip']}")
    print("  a. ALL of the above")
    print("===================================")
    print(" Enter numbers separated by commas (e.g. 1,3,4)")
    print(" or 'a' for all.")

    while True:
        choice = input("\nYour selection: ").strip().lower()

        if choice in ("a", "all"):
            return list(KNOWN_LAPTOPS)

        # parse comma-separated numbers
        try:
            picks = [int(x) for x in choice.replace(" ", "").split(",") if x]
        except ValueError:
            print("  Invalid input. Use numbers like 1,3 or 'a'.")
            continue

        valid = [n for n in picks if 1 <= n <= len(KNOWN_LAPTOPS)]
        if not valid:
            print("  No valid numbers. Try again.")
            continue

        selected = [KNOWN_LAPTOPS[n - 1] for n in valid]
        print("\nSelected:")
        for lap in selected:
            print(f"  - {lap['name']} ({lap['ip']})")
        return selected


# ==========================
# Discovery for the chosen laptops
# ==========================
def discover_sources(laptops):
    """Query each chosen laptop's /list and return [(url, name), ...] for ALL cameras."""
    sources = []
    print("\nQuerying chosen laptops...")
    for lap in laptops:
        ip, label = lap["ip"], lap["name"]
        try:
            r = requests.get(f"http://{ip}:{PORT}/list", timeout=TIMEOUT)
            data = r.json()
        except Exception as e:
            print(f"  [{label} / {ip}] unreachable: {e}")
            continue

        cams = data.get("cameras", [])
        if not cams:
            print(f"  [{label} / {ip}] no cameras reported")
            continue

        for c in cams:
            url = c["url"]
            # friendly, unique name: "Floor-1 Laptop-cam0"
            name = f"{label}-cam{c['index']}"
            sources.append((url, name))
            print(f"  [{label}] {name} -> {url}")

    print(f"\nTotal camera streams: {len(sources)}")
    return sources


# ==========================
# Convenience: choose + discover in one call (import this from your YOLO script)
# ==========================
def get_sources():
    laptops = choose_laptops()
    return discover_sources(laptops)


def discover_all():
    """Non-interactive: query EVERY known laptop and return all their cameras.
    Used by the dashboard 'Search laptops' button and by startup auto-connect."""
    return discover_sources(KNOWN_LAPTOPS)


if __name__ == "__main__":
    sources = get_sources()

    print("\n===================================")
    print(" Final Camera Source List")
    print("===================================")
    for url, name in sources:
        print(f"  {name}: {url}")

    # ---- Hand these to your CameraStream + YOLO pipeline ----
    # from camera_helper import CameraStream
    # streams = [CameraStream(url, name=name).start() for url, name in sources]
    #
    # while True:
    #     for s in streams:
    #         ok, frame = s.read()
    #         if ok:
    #             results = model(frame, conf=0.25, verbose=False, device=0)
    #             ... # plot, log, alert using s.name and s.healthy()