# Windows RealtimeSTT + Ollama Setup

Minimal setup for running speech transcription (RealtimeSTT) and LLMs (Ollama) on Windows with NVIDIA GPU.

**Last updated:** December 2025

---

## Prerequisites

### Check GPU
```powershell
(Get-CimInstance Win32_VideoController).Name
nvidia-smi
```

### Install NVIDIA Driver
1. [nvidia.com/drivers](https://nvidia.com/drivers) → Download **Studio Driver**
2. Install with **"Perform clean installation"** checked

---

## Step 1: Install `uv` (Python Manager)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart terminal.

---

## Step 2: Install Python 3.11

```powershell
uv python install 3.11 --default
python --version  # Should show 3.11.x
```

---

## Step 3: Install RealtimeSTT

```powershell
mkdir C:\code\stt
cd C:\code\stt
uv venv
.venv\Scripts\activate
uv pip install RealtimeSTT
```

**Test:**
```powershell
python -c "from RealtimeSTT import AudioToTextRecorder; print('Ready!')"
```

---

## Step 4: Install Ollama

1. Download from [ollama.com](https://ollama.com)
2. Run installer
3. Ollama auto-detects GPU

**Test:**
```powershell
ollama run llama3.2
```

---

## If Ollama Doesn't Use GPU

### Fix 1: Windows Graphics Settings
1. Win+S → "Graphics Settings"
2. Browse → `C:\Users\<You>\AppData\Local\Programs\Ollama\ollama app.exe`
3. Add → Options → **High performance**

### Fix 2: Environment Variable
1. Win+S → "Edit system environment variables"
2. Environment Variables → System variables → New:
   - Variable: `CUDA_VISIBLE_DEVICES`
   - Value: `0`
3. Restart Ollama

### Verify GPU Usage
```powershell
# While running ollama:
nvidia-smi
# Should show ollama process
```

---

## Recommended Versions (Dec 2025)

| Component | Version |
|-----------|---------|
| Python | 3.11.x |
| NVIDIA Driver | 573.xx+ |
| RealtimeSTT | latest |
| Ollama | latest |

