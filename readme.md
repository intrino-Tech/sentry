# =========================================================================
# SENTRY — fire/smoke detection  |  requirements.txt
# =========================================================================
# Target platform: Jetson Orin Nano
#   JetPack 7.2 / Jetson Linux R39.2 / CUDA 13.2 / Python 3.12 / sm_87
#
# INSTALL — the order matters, and torch must come from the CUDA index.
#
#   python3 -m venv --system-site-packages .venv
#   source .venv/bin/activate
#   pip install --upgrade pip
#   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu132
#
# --system-site-packages is REQUIRED: TensorRT ships with JetPack into the
# system dist-packages and is not pip-installable on this platform. A venv
# without that flag cannot see it and every .engine load will fail.
#
# --extra-index-url (NOT --index-url) is REQUIRED: the plain PyPI torch wheel
# is built for sm_110/sm_121 datacenter GPUs and dies with "no kernel image is
# available for execution on the device" on Orin's sm_87. Using --index-url
# instead would also stop numpy and the rest resolving from PyPI.
#
# VERIFY before running anything:
#   python3 -c "import tensorrt; print(tensorrt.__version__)"
#   python3 -c "import torch; x=torch.randn(64,64,device='cuda'); print((x@x).sum().item())"
# =========================================================================

# ---- Deep learning runtime -------------------------------------------------
# Pinned loosely: the CUDA suffix, not the version, is what matters here.
# If you change JetPack, change the --extra-index-url suffix to match the new
# CUDA version (cu130 / cu131 / cu132 ...) and rebuild the .engine files.
torch
torchvision

# tensorrt is NOT listed: it comes from JetPack via --system-site-packages.
# Do not pip install it — the PyPI wheel is for x86 and will shadow the
# working system copy.

# ---- Detection -------------------------------------------------------------
ultralytics>=8.3.0

# ---- Vision / IO -----------------------------------------------------------
# NOTE ON GSTREAMER: the pip wheel below is built WITHOUT GStreamer support,
# so RTSP decode falls back to FFmpeg on the CPU (tegrastats will show
# "NVDEC0 off"). To use the Orin's hardware decoder, either drop this line and
# rely on JetPack's system OpenCV (visible through --system-site-packages), or
# build OpenCV from source with -D WITH_GSTREAMER=ON.
#   Check which one you have:  python3 -c "import cv2; print(cv2.__file__)"
#   Check GStreamer support:   python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer
opencv-python>=4.8.0

numpy<2.0.0
Pillow

# ---- Web / UI --------------------------------------------------------------
Flask>=3.0.0
Werkzeug>=3.0.0          # password hashing for auth.py (Flask dep, pinned explicitly)

# ---- Reporting -------------------------------------------------------------
reportlab>=4.0.0

# ---- Sensors (optional) ----------------------------------------------------
# Only needed if you attach a SensorHub to the Detector. Harmless otherwise.
paho-mqtt>=2.0.0

# ---- Production server -----------------------------------------------------
# Flask's built-in server is single-threaded and warns against production use.
# Run under waitress instead:
#   waitress-serve --host=0.0.0.0 --port=8000 --call run:make_app
waitress>=3.0.0

# =========================================================================
# EXPORT-ONLY — needed to build .engine files, not to run the service
# =========================================================================
# Install these only when rebuilding engines:
#   pip install onnx onnxslim
#
# Do NOT add onnxruntime-gpu: it has no aarch64 wheel on PyPI, and because pip
# installs atomically, including it makes the whole install fail — which is
# what silently blocked the first TensorRT export attempt. It is not needed
# for exporting, only for running ONNX inference, which this project does not.
#
# onnx>=1.12.0,<2.0.0
# onnxslim>=0.1.82