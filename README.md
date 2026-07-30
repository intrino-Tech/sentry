# Sentry AI Surveillance System

An AI-powered surveillance dashboard for real-time fire and smoke monitoring using YOLO object detection. The system provides a web interface to monitor multiple cameras, detect safety events, generate reports, and maintain event logs.

---

# Features

- Real-time AI camera monitoring
- Multi-camera support
- YOLO-based object detection
- Fire & Smoke detection
- Dashboard with live statistics
- Event logging
- Report generation (PDF)
- User authentication
- Flask-based web interface

---

# Project Structure

```
.
├── run.py                  # Main entry point
├── web_app.py              # Flask web application
├── detector.py             # Detection pipeline
├── camera_helper.py        # Camera utilities
├── main_pc_sources.py      # Camera source configuration
├── auth.py                 # Authentication
├── report.py               # PDF report generation
├── config.py               # Configuration
├── requirements.txt
├── install.bat             # Automatic dependency installer
├── templates/
├── best.pt                 # Detection model
├── firebest.pt             # Fire detection model
└── users.json
```

---

# System Requirements

## Operating System

- Windows 10 / Windows 11 (Recommended)

## Python

Python 3.10 or newer

Download:

https://www.python.org/downloads/

During installation, **enable**:

- ✔ Add Python to PATH

---

# Hardware Requirements

Minimum

- Intel i5 Processor
- 8 GB RAM
- NVIDIA GPU (optional)

Recommended

- Intel i7 / Ryzen 7
- 16 GB RAM
- NVIDIA RTX GPU with CUDA

---

# Initial Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

or simply download and extract the ZIP.

---

## 2. Create a Virtual Environment

Windows

```bash
python -m venv venv
```

Activate

Command Prompt

```cmd
venv\Scripts\activate
```

PowerShell

```powershell
venv\Scripts\Activate.ps1
```

---

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

# Automatic Installation (Recommended)

The project already includes an installer.

Simply run

```cmd
install.bat
```

The installer will automatically:

- Upgrade pip
- Install CUDA-enabled PyTorch
- Install all required Python packages
- Verify CUDA installation
- Display GPU information

---

# Manual Installation

If you prefer manual installation,

Install CUDA-enabled PyTorch

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

Install remaining packages

```bash
pip install -r requirements.txt
```

---

# Verify Installation

Run

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output

```
True
```

If using CPU only, it may return

```
False
```

---

# Required Models

Place the trained model files in the project root.

```
best.pt
firebest.pt
```

---

# Running the Application

Start the application

```bash
python run.py
```

The terminal will display the local server URL.

Example

```
http://127.0.0.1:5000
```

Open the URL in your browser.

---

# Camera Configuration

Camera sources are configured inside

```
main_pc_sources.py
```

Modify the camera URLs or RTSP streams as required.

Example

```python
CAMERA_1 = "rtsp://username:password@ip-address/stream"
```

---

# Dashboard

The dashboard provides

- Live video feed
- Detection status
- Event history
- Camera status
- Reports
- Authentication

---

# Reports

Reports are automatically generated using

```
report.py
```

Generated reports include

- Detection summary
- Timestamp
- Event details

---

# Authentication

User information is stored in

```
users.json
```

Authentication logic is implemented in

```
auth.py
```

---

# Updating Dependencies

Whenever new packages are added,

Update the requirements file

```bash
pip freeze > requirements.txt
```

Other users can update using

```bash
pip install -r requirements.txt
```

---

# Common Issues

## ModuleNotFoundError

Install missing packages

```bash
pip install -r requirements.txt
```

---

## CUDA Not Available

Verify installation

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`

- Check NVIDIA drivers
- Verify CUDA installation
- Reinstall CUDA-enabled PyTorch

---

## Camera Not Opening

- Verify RTSP URL
- Ensure camera is reachable
- Check firewall settings
- Verify network connectivity

---

## Model Not Found

Ensure the following files exist in the project root

```
best.pt
firebest.pt
```

---

# Stopping the Application

Press

```
CTRL + C
```

inside the terminal.

---

# Technologies Used

- Python
- Flask
- Ultralytics YOLO
- OpenCV
- PyTorch
- NumPy
- Pillow
- ReportLab

---

# License

This project is intended for internal use unless otherwise specified.
=======
# sentry
>>>>>>> 816d6ad61c698d27825add25da4a34fbe0d2f2b8
