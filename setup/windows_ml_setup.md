# Windows ML/Transcription Setup Guide

Setup guide for running LLMs (Ollama), speech transcription (Whisper, RealtimeSTT), and ML workloads on Windows with NVIDIA GPU.

**Last updated:** December 2025

---

## Prerequisites Check

### Identify GPUs (PowerShell)
```powershell
Get-CimInstance Win32_VideoController | Select-Object Name
# Or cleaner output:
(Get-CimInstance Win32_VideoController).Name
```

### Check NVIDIA Driver
```powershell
nvidia-smi
```
- **Driver Version:** Should be 573.xx+ (or latest)
- **CUDA Version:** Should be 12.6+ or 13.x

---

## Step 1: Install NVIDIA Driver (if needed)

1. Go to [nvidia.com/drivers](https://nvidia.com/drivers)
2. Select your GPU model
3. Download **Studio Driver** (more stable for ML than Game Ready)
4. Run installer → **Custom (Advanced)** → ✅ **"Perform clean installation"**
5. Restart PC

---

## Step 2: Install `uv` (Python Manager)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart terminal after install.

---

## Step 3: Install Python 3.11

**Why 3.11?** RealtimeSTT requires `<3.13`. Python 3.11 has best compatibility.

```powershell
# Install and set as default
uv python install 3.11 --default

# Verify
python --version  # Should show 3.11.x

# Optional: install multiple versions
uv python install 3.10 3.11 3.12
```

### Useful `uv` Commands
| Goal | Command |
|------|---------|
| List installed versions | `uv python list` |
| Run with specific version | `uv run --python 3.10 script.py` |
| Uninstall a version | `uv python uninstall 3.10` |

---

## Step 4: Install CUDA Toolkit + cuDNN (if needed)

**Note:** Usually not required if you let PyTorch bundle its own CUDA runtime.

If you need system-wide CUDA:

1. **CUDA 12.6:** Download from [developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)
2. **cuDNN 9.x:** Download from [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
3. Copy cuDNN files:
   ```
   bin\*.dll    → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin
   include\*.h  → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include
   lib\x64\*.lib → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64
   ```
4. **zlib fix:** Download [zlib123dllx64.zip](http://www.winimage.com/zLibDll/zlib123dllx64.zip), copy `zlibwapi.dll` to CUDA bin folder

---

## Step 5: Install PyTorch

For CUDA 12.6+ (compatible with most drivers):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Verify GPU Detection
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## Step 6: Install RealtimeSTT

```powershell
pip install RealtimeSTT
pip install --upgrade ctranslate2>=4.5.0
```

### Verify
```python
from RealtimeSTT import AudioToTextRecorder
recorder = AudioToTextRecorder(model="tiny.en", device="cuda")
print("RealtimeSTT initialized on GPU!")
```

---

## Step 7: Install & Configure Ollama

### Install Ollama
Download from [ollama.com](https://ollama.com)

### Force Ollama to Use GPU

**Fix 1: Windows Graphics Settings**
1. Win+S → "Graphics Settings"
2. Browse → `C:\Users\<YourUser>\AppData\Local\Programs\Ollama\ollama app.exe`
3. Add → Options → **High performance** (NVIDIA GPU)

**Fix 2: Environment Variables**
1. Win+S → "Edit the system environment variables"
2. Environment Variables → System variables → New:

| Variable | Value |
|----------|-------|
| `CUDA_VISIBLE_DEVICES` | `0` |
| `OLLAMA_LLM_LIBRARY` | `cuda_v12` |

3. Restart Ollama (quit from system tray, relaunch)

**Fix 3: Clean Driver Install**
If still not working, reinstall NVIDIA driver with "Perform clean installation" checked.

### Verify Ollama GPU Usage
```powershell
ollama run llama3.2
# In another terminal:
nvidia-smi
# Should show ollama process using GPU memory
```

### Check Ollama Logs
```
%LOCALAPPDATA%\Ollama\server.log
```
- ✅ Success: `detected GPUs ... library=cuda ... total="xx GiB"`
- ❌ Failure: `CUDA GPU is too old` or `unable to load cudart library`

---

## Python Version Compatibility (Dec 2025)

| Library | 3.10 | 3.11 | 3.12 | 3.13 |
|---------|:----:|:----:|:----:|:----:|
| **Ollama** | ✅ | ✅ | ✅ | ✅ |
| **Whisper/faster-whisper** | ✅ | ✅ | ✅ | ⚠️ |
| **RealtimeSTT** | ✅ | ✅ | ✅ | ❌ |
| **PyTorch** | ✅ | ✅ | ✅ | ✅ |

**Recommended: Python 3.11**

---

## Recommended Versions (Dec 2025)

| Component | Version |
|-----------|---------|
| Python | 3.11.x |
| NVIDIA Driver | 573.xx+ (Studio) |
| CUDA (via PyTorch) | 12.6 |
| cuDNN | 9.x |
| PyTorch | 2.9.x |
| CTranslate2 | ≥4.5.0 |

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `cublas64_12.dll not found` | CUDA not in PATH | Add CUDA bin to PATH |
| `Could not locate zlibwapi.dll` | Missing zlib | Copy zlibwapi.dll to CUDA bin |
| `nvidia-smi` not recognized | Driver not installed | Install NVIDIA driver |
| Ollama not using GPU | Windows defaulting to iGPU | Fix 1, 2, or 3 above |
| `OMP: Error #15` | Library conflict | Add `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` |

