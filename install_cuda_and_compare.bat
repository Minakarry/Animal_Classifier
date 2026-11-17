@echo off
cd /d D:\wheels
echo Fresh download torch 2.4.1+cu118...
curl.exe -L --retry 30 --retry-all-errors --ssl-no-revoke -o torch-2.4.1+cu118-cp310-cp310-win_amd64.whl "https://download.pytorch.org/whl/cu118/torch-2.4.1%%2Bcu118-cp310-cp310-win_amd64.whl"
if errorlevel 1 exit /b 1
echo Downloading torchvision...
curl.exe -L --retry 30 --retry-all-errors --ssl-no-revoke -o torchvision-0.19.1+cu118-cp310-cp310-win_amd64.whl "https://download.pytorch.org/whl/cu118/torchvision-0.19.1%%2Bcu118-cp310-cp310-win_amd64.whl"
if errorlevel 1 exit /b 1
for %%F in (torch-2.4.1+cu118-cp310-cp310-win_amd64.whl) do echo torch size=%%~zF
set TEMP=D:\pip_tmp
set TMP=D:\pip_tmp
echo Installing...
D:\??\PYTHON~1\python.exe -m pip install --force-reinstall "D:\wheels\torch-2.4.1+cu118-cp310-cp310-win_amd64.whl" "D:\wheels\torchvision-0.19.1+cu118-cp310-cp310-win_amd64.whl"
if errorlevel 1 exit /b 1
D:\??\PYTHON~1\python.exe -c "import torch; print('VER', torch.__version__); print('CUDA', torch.cuda.is_available()); print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
cd /d E:\Animal_Classifier
echo Starting compare.py...
D:\??\PYTHON~1\python.exe compare.py --epochs 10
