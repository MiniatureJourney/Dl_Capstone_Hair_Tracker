@echo off
REM ============================================================
REM  fix_environment.bat
REM  Resolves all dependency conflicts for the capstone project
REM  Run from: DL Capstone\DL_Capstone_HairSegmentation\
REM ============================================================

SET PYTHON=C:\Users\Aliasgar\Desktop\DL Capstone\capstone_env\Scripts\python.exe

echo [1/5] Upgrading pip...
"%PYTHON%" -m pip install --upgrade pip --quiet

echo [2/5] Upgrading numpy to 2.x (required by scipy + torchmetrics)...
"%PYTHON%" -m pip install "numpy>=2.0,<3.0" --upgrade --quiet

echo [3/5] Reinstalling scipy compatible version...
"%PYTHON%" -m pip install scipy --upgrade --quiet

echo [4/5] Installing remaining packages...
"%PYTHON%" -m pip install opencv-python Pillow scikit-image albumentations torchmetrics customtkinter tqdm matplotlib pandas seaborn fpdf2 jupyter ipywidgets --upgrade --quiet

echo [5/5] Verifying environment...
"%PYTHON%" src\verify_setup.py

echo.
echo ============================================================
echo   Environment fix complete!
echo ============================================================
pause
