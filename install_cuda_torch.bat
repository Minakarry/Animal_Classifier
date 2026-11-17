@echo off
set TEMP=D:\pip_tmp
set TMP=D:\pip_tmp
set PIP_CACHE_DIR=D:\pip_cache
echo Downloading CUDA torch wheel to D:\wheels (resume supported)...
curl.exe -L --retry 20 --retry-all-errors --ssl-no-revoke -C - -o "D:\wheels\torch-2.7.1+cu118-cp310-cp310-win_amd64.whl" "https://download.pytorch.org/whl/cu118/torch-2.7.1%%2Bcu118-cp310-cp310-win_amd64.whl"
if errorlevel 1 exit /b 1
curl.exe -L --retry 20 --retry-all-errors --ssl-no-revoke -C - -o "D:\wheels\torchvision-0.22.1+cu118-cp310-cp310-win_amd64.whl" "https://download.pytorch.org/whl/cu118/torchvision-0.22.1%%2Bcu118-cp310-cp310-win_amd64.whl"
if errorlevel 1 exit /b 1
echo Installing...
"D:\??\Python??\python.exe" -m pip install --force-reinstall "D:\wheels\torch-2.7.1+cu118-cp310-cp310-win_amd64.whl" "D:\wheels\torchvision-0.22.1+cu118-cp310-cp310-win_amd64.whl"
"D:\??\Python??\python.exe" -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
