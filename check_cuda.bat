@echo off
echo ====================================================================
echo CUDA Verification Tool for CritterDetector
echo ====================================================================
echo.

echo Checking CUDA availability...
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to import PyTorch. Please ensure it's installed correctly.
    goto :end
)

python -c "import torch; is_available = torch.cuda.is_available(); print(f'CUDA device count: {torch.cuda.device_count() if is_available else 0}')"
python -c "import torch; is_available = torch.cuda.is_available(); print(f'CUDA device name: {torch.cuda.get_device_name(0) if is_available and torch.cuda.device_count() > 0 else \"None\"}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda if hasattr(torch.version, \"cuda\") else \"Not found\"}')"

echo.
echo Checking NVIDIA driver...
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo NVIDIA driver information:
    nvidia-smi
) else (
    echo Warning: nvidia-smi not found in PATH. Cannot display driver information.
)

echo.
echo Checking for CUDA Toolkit (nvcc)...
where nvcc >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo CUDA Toolkit found:
    nvcc --version
) else (
    echo Warning: CUDA Toolkit (nvcc) not found in PATH.
    echo This may be okay if you're using pre-built CUDA binaries through PyTorch.
)

echo.
echo Checking installed models...
python -c "from transformers import Owlv2Processor; print('OWLv2 available')" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo - OWLv2: Available
) else (
    echo - OWLv2: Not available
)

python -c "from transformers import DetrImageProcessor; print('DETR available')" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo - DETR: Available
) else (
    echo - DETR: Not available
)

python -c "from ultralytics import YOLO; print('YOLO available')" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo - YOLO: Available
) else (
    echo - YOLO: Not available
)

echo.
echo Running quick CUDA test...
python -c "import torch; x = torch.zeros(1000,1000); y = torch.matmul(x, x); cuda_works = torch.cuda.is_available(); x_cuda = x.cuda() if cuda_works else x; y_cuda = torch.matmul(x_cuda, x_cuda) if cuda_works else y; print(f'CUDA matrix multiplication test: {\"Success\" if cuda_works else \"Skipped - CUDA not available\"}')"

echo.
echo ====================================================================
echo If CUDA is not available but you have an NVIDIA GPU:
echo 1. Install the NVIDIA CUDA Toolkit
echo 2. Install PyTorch with CUDA support using:
echo    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo    (Replace cu118 with your CUDA version)
echo ====================================================================

:end
pause 