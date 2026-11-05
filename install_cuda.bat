@echo off
REM ============================================================
REM  install_cuda.bat
REM  Installs PyTorch 2.7 + CUDA 12.8 for RTX 5060 (Blackwell)
REM  Run ONCE after activating capstone_env
REM ============================================================

echo [STEP 1] Installing PyTorch 2.7.0 with CUDA 12.8 for RTX 5060...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo [STEP 2] Installing remaining dependencies...
pip install -r requirements.txt

echo [STEP 3] Verifying CUDA availability...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND')"

echo.
echo ============================================================
echo  If you see your GPU name above, setup is COMPLETE!
echo  Run next: python src/verify_setup.py
echo ============================================================
pause
