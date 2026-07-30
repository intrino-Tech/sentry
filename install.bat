@echo off
echo ======================================
echo Installing Project Dependencies
echo ======================================

python -m pip install --upgrade pip

echo.
echo Installing PyTorch (CUDA 12.8)...
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio

echo.
echo Installing remaining requirements...
pip install -r requirements.txt

echo.
echo Verifying CUDA installation...
python -c "import torch; print('Torch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU Found')"

echo.
echo ======================================
echo Installation Complete
echo ======================================

pause