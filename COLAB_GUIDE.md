# Google Colab Quick Start Guide

## Setup (5 minutes)

### 1. Enable GPU
```
Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save
```

### 2. Clone & Navigate
```python
!git clone https://github.com/YOUR_USERNAME/Medical_AI_Agent.git
%cd Medical_AI_Agent/Codebase
```

### 3. Install Dependencies
```python
!pip install -q -r requirements.txt
```

### 4. Start Ollama (Background)
```python
import subprocess
import time

# Start Ollama service
ollama_process = subprocess.Popen(['ollama', 'serve'])
time.sleep(5)

# Pull Llama model
!curl -fsSL https://ollama.com/install.sh | sh
!ollama pull llama3.2
```

---

## Interaction Options

### Option 1: Gradio UI (Recommended)

**Automatic public URL - easiest way to use in Colab!**

```python
!python main.py --ui
```

**Output:**
```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live  ← Use this URL
```

**Features:**
- Click the public URL to open UI in new tab
- Upload images, paste text, get results
- Beautiful interface with parsing
- Session management
- URL valid for 72 hours

---

### Option 2: Python API (Programmatic)

**For notebook-based analysis:**

```python
from cli_main import MedicalAssistantOrchestrator
from models import MedicalText, MedicalImage, ImageModality

# Initialize
orchestrator = MedicalAssistantOrchestrator()

# Analyze text
result = orchestrator.analyze_text(
    MedicalText(
        text="Patient presents with chest pain...",
        text_type="clinical_note"
    )
)
print(result.analysis)

# Analyze image (upload first)
from google.colab import files
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

result = orchestrator.analyze_image(
    MedicalImage(
        image_path=image_path,
        modality=ImageModality.XRAY,
        body_part="chest"
    )
)
print(result.findings)
```

---

### Option 3: CLI Mode (Interactive)

**NOT recommended for Colab** - use Gradio UI instead. CLI requires user input which doesn't work well in notebooks.

---

## Complete Notebook

Use `Colab_Setup.ipynb` - ready-to-run notebook with all steps!

**Just upload to Colab and run cells sequentially.**

---

## File Upload/Download in Colab

### Upload Files
```python
from google.colab import files

# Upload image
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
```

### Download Results
```python
# Download output file
files.download('output.json')
```

---

## Colab-Specific Fixes

### Issue: Virtual Environment Error

**Fixed!** The code now detects Colab automatically:

```python
# In main.py
if self.is_colab:
    print_info("Running in Google Colab - using system Python")
    # Skips venv creation
```

### Verification
```python
# Check if Colab detected
import sys
print(f"Python: {sys.executable}")
# Should show: /usr/bin/python3 (system Python)
```

---

## Performance on T4 GPU

| Component | Device | Time |
|-----------|--------|------|
| BioBERT | CUDA | ~20ms |
| BiomedCLIP | CPU* | ~8s |
| Llama 3 | CUDA | ~2s |

**Note:** BiomedCLIP currently hardcoded to CPU. To enable GPU:

```python
# Edit Codebase/image_processor.py line 63
# Change:
self.device = torch.device("cpu")

# To:
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Then restart runtime and rerun.

---

## Common Issues

### 1. "Ollama connection refused"
```python
# Restart Ollama
!killall ollama
!ollama serve &
!sleep 5
!ollama pull llama3.2
```

### 2. "CUDA out of memory"
```
Runtime → Factory reset runtime
Then rerun from start
```

### 3. "Gradio not showing URL"
```python
# Check Gradio version
!pip install --upgrade gradio

# Restart runtime
Runtime → Restart runtime
```

### 4. "Module not found"
```python
# Verify you're in correct directory
%cd /content/Medical_AI_Agent/Codebase
!pwd

# Reinstall dependencies
!pip install -r requirements.txt
```

---

## Quick Test Script

```python
# Verify everything works
!python test_cuda_devices.py

# Expected output:
# [1/6] PyTorch Device Availability: ✓
# [2/6] BioBERT: ✓
# [3/6] BiomedCLIP: ✓
# [4/6] FAISS: ✓
# [5/6] Ollama: ✓
# [6/6] Application Modules: ✓
```

---

## Session Management

**Colab sessions timeout after:**
- 90 minutes idle
- 12 hours maximum (free tier)

**To preserve work:**
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Save outputs to Drive
!cp -r output/ /content/drive/MyDrive/medical_ai_outputs/
```

---

## Best Practices

1. **Always enable GPU** - 10x faster inference
2. **Use Gradio UI** - easiest interaction in Colab
3. **Save to Drive** - sessions expire
4. **Monitor memory** - `!nvidia-smi` to check GPU
5. **Use public URL** - share with others (valid 72h)

---

## Summary

**Fastest way to get started:**

1. Enable T4 GPU in runtime settings
2. Upload `Colab_Setup.ipynb` to Colab
3. Run all cells
4. Click Gradio public URL
5. Start analyzing!

**Total setup time: 5-10 minutes**
