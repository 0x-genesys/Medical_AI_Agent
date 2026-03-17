# Model Device Usage Analysis

## Current Device Configuration

### Models in Your Codebase

| Model | Component | Current Device Logic | Your Laptop (MPS) | CUDA Behavior |
|-------|-----------|---------------------|-------------------|---------------|
| **BioBERT (Bio_ClinicalBERT)** | TextProcessor | Auto-detect: CUDA → MPS → CPU | **MPS (Apple GPU)** | Would use CUDA |
| **BiomedCLIP** | ImageProcessor | **Hardcoded CPU** | **CPU only** | CPU (needs change) |
| **Llama 3 (8B)** | OllamaLLM | Ollama manages device | **GPU via Ollama** | GPU via Ollama |
| **FAISS Index** | TextProcessor | CPU always | **CPU** | CPU (default) |

---

## Detailed Analysis

### 1. BioBERT (Text Embeddings)

**File**: `text_processor.py:35-47`

```python
# Detect GPU for BioBERT embeddings
if torch.cuda.is_available():
    self.device = "cuda"
    self.logger.info(f"BioBERT will use NVIDIA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    self.device = "mps"
    self.logger.info("BioBERT will use Apple Silicon GPU (MPS)")
else:
    self.device = "cpu"
    self.logger.warning("BioBERT using CPU (no GPU detected)")

self.embedder = SentenceTransformer(embedding_model, device=self.device)
```

**Current behavior on your MPS laptop**: Uses Apple Silicon GPU (MPS)  
**On CUDA machine**: Would automatically use CUDA GPU  
**Performance**: MPS gives 3-5x speedup vs CPU, CUDA would give 5-10x

---

### 2. BiomedCLIP (Vision Model)

**File**: `image_processor.py:63`

```python
# Device selection (CPU by default, MPS/CUDA if available)
self.device = torch.device("cpu")  # ← HARDCODED TO CPU
```

**Current behavior**: **Always CPU**, even with GPU available  
**On CUDA machine**: Still CPU (needs code change)  
**Performance impact**: ~8 seconds per image (could be <1s on GPU)

**To enable GPU** (MPS or CUDA):
```python
# Option 1: Auto-detect
if torch.cuda.is_available():
    self.device = torch.device("cuda")
elif torch.backends.mps.is_available():
    self.device = torch.device("mps")
else:
    self.device = torch.device("cpu")

# Option 2: Force specific device
self.device = torch.device("cuda")  # For CUDA testing
```

---

### 3. Llama 3 (Large Language Model)

**File**: `text_processor.py:49`, `image_processor.py:89`

```python
self.llm = OllamaLLM(model=self.model_name, temperature=config.model.temperature)
```

**Device management**: Handled by Ollama, not Python code  
**Current behavior**: Ollama automatically uses GPU if available  
**On your MPS laptop**: Uses Metal (Apple's GPU API)  
**On CUDA machine**: Would use CUDA automatically

**Verify Ollama device usage**:
```bash
# Check if Ollama is using GPU
ollama ps  # Shows running models and memory usage

# Force CPU (if needed for testing)
OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 ollama serve

# Check Ollama GPU usage
nvidia-smi  # On CUDA systems
```

---

### 4. FAISS (Vector Search)

**File**: `text_processor.py` (FAISS index operations)

**Device**: Always CPU (FAISS-CPU package)  
**GPU version**: Available as `faiss-gpu` but not installed  
**Performance**: CPU is sufficient for current knowledge base size

**To enable GPU** (requires CUDA, not MPS):
```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

Note: FAISS GPU requires CUDA, does NOT support MPS.

---

## Device Usage Summary

### On Your MPS Laptop (Current)

```
┌─────────────────────────────────────┐
│   Model Device Distribution          │
├─────────────────────────────────────┤
│ BioBERT        → MPS (Apple GPU)    │
│ BiomedCLIP     → CPU (hardcoded)    │
│ Llama 3        → Metal (via Ollama) │
│ FAISS          → CPU (always)       │
└─────────────────────────────────────┘
```

### On CUDA Machine (Expected)

```
┌─────────────────────────────────────┐
│   Model Device Distribution          │
├─────────────────────────────────────┤
│ BioBERT        → CUDA (auto)        │
│ BiomedCLIP     → CPU (needs fix)    │
│ Llama 3        → CUDA (via Ollama)  │
│ FAISS          → CPU (or GPU if pkg)│
└─────────────────────────────────────┘
```

---

## Testing on CUDA from MPS Laptop

### Option 1: Cloud GPU Instance (Recommended)

**Free/cheap options:**
1. **Google Colab** (Free T4 GPU)
   ```bash
   # Upload your code to Colab
   # Run in notebook cell:
   !git clone <your-repo>
   !cd Codebase && pip install -r requirements.txt
   !python main.py --cli
   ```

2. **Kaggle Notebooks** (Free P100 GPU, 30h/week)
   - Similar to Colab
   - Better GPU quota

3. **AWS SageMaker Studio Lab** (Free GPU, registration required)
   - 4 hours GPU sessions
   - ml.g4dn.xlarge (T4 GPU)

4. **Paperspace Gradient** (Free tier with GPU)
   - C3 or C4 instances

**Best for testing**: Google Colab (easiest setup)

---

### Option 2: Docker + Remote GPU

Use Docker to test locally, deploy to CUDA cloud:

**Dockerfile for CUDA**:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy code
COPY . .

# Pre-download models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')"

CMD ["python", "main.py", "--cli"]
```

**Deploy to cloud with GPU**:
```bash
# Build
docker build -t medical-assistant-cuda .

# Run on cloud (AWS, GCP, Azure)
docker run --gpus all -p 7860:7860 medical-assistant-cuda
```

---

### Option 3: SSH into CUDA Machine

If you have access to a CUDA workstation/server:

```bash
# From your MPS laptop
# 1. Copy code to CUDA machine
scp -r Codebase/ user@cuda-machine:/path/to/

# 2. SSH into CUDA machine
ssh user@cuda-machine

# 3. Setup on CUDA machine
cd /path/to/Codebase
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Verify CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA device:', torch.cuda.get_device_name(0))"

# 5. Run tests
python main.py --cli
```

---

### Option 4: GitHub Codespaces (If you have access)

GitHub Codespaces can provision GPU instances:

```yaml
# .devcontainer/devcontainer.json
{
  "name": "Medical Assistant CUDA",
  "image": "pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
  "hostRequirements": {
    "gpu": true
  },
  "features": {
    "ghcr.io/devcontainers/features/python:1": {}
  },
  "postCreateCommand": "pip install -r requirements.txt"
}
```

---

## Quick CUDA Testing Script

Create this script to verify CUDA setup:

**File**: `test_cuda_devices.py`

```python
#!/usr/bin/env python3
"""Test script to verify device availability and model loading on CUDA"""

import torch
import sys

print("="*60)
print("CUDA Device Testing")
print("="*60)

# 1. Check PyTorch CUDA
print("\n[1/5] PyTorch CUDA:")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Device count: {torch.cuda.device_count()}")
    print(f"  Current device: {torch.cuda.current_device()}")
    print(f"  Device name: {torch.cuda.get_device_name(0)}")
    print(f"  Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("  ❌ CUDA not available")
    print(f"  MPS available: {torch.backends.mps.is_available()}")

# 2. Test BioBERT
print("\n[2/5] Testing BioBERT:")
try:
    from sentence_transformers import SentenceTransformer
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Loading on: {device}")
    
    model = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT', device=device)
    embedding = model.encode("Test sentence")
    print(f"  ✓ BioBERT working on {device}")
    print(f"  Embedding shape: {embedding.shape}")
except Exception as e:
    print(f"  ❌ BioBERT failed: {e}")

# 3. Test BiomedCLIP
print("\n[3/5] Testing BiomedCLIP:")
try:
    from open_clip import create_model_from_pretrained
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Loading on: {device}")
    
    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    model.to(device)
    model.eval()
    print(f"  ✓ BiomedCLIP working on {device}")
except Exception as e:
    print(f"  ❌ BiomedCLIP failed: {e}")

# 4. Test FAISS
print("\n[4/5] Testing FAISS:")
try:
    import faiss
    print(f"  FAISS version: {faiss.__version__}")
    
    # Check if GPU version
    if hasattr(faiss, 'get_num_gpus'):
        gpu_count = faiss.get_num_gpus()
        print(f"  GPU support: Yes ({gpu_count} GPUs)")
    else:
        print(f"  GPU support: No (CPU-only)")
    
    # Test index creation
    index = faiss.IndexFlatL2(768)
    print(f"  ✓ FAISS working")
except Exception as e:
    print(f"  ❌ FAISS failed: {e}")

# 5. Test Ollama
print("\n[5/5] Testing Ollama:")
try:
    import subprocess
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ Ollama installed")
        print(f"  Models: {result.stdout.strip()}")
    else:
        print(f"  ❌ Ollama not working")
except Exception as e:
    print(f"  ❌ Ollama failed: {e}")

print("\n" + "="*60)
print("Device Testing Complete")
print("="*60)
```

**Run on CUDA machine**:
```bash
python test_cuda_devices.py
```

---

## Enabling BiomedCLIP on GPU

To enable BiomedCLIP GPU acceleration, update `image_processor.py`:

**Current** (line 63):
```python
self.device = torch.device("cpu")  # Hardcoded
```

**Updated** (auto-detect):
```python
# Auto-detect best device
if torch.cuda.is_available():
    self.device = torch.device("cuda")
    self.logger.info("BiomedCLIP will use CUDA GPU")
elif torch.backends.mps.is_available():
    self.device = torch.device("mps")
    self.logger.info("BiomedCLIP will use Apple Silicon GPU (MPS)")
else:
    self.device = torch.device("cpu")
    self.logger.warning("BiomedCLIP using CPU (no GPU detected)")
```

**Or force CUDA** (for testing):
```python
# Force CUDA for testing
self.device = torch.device("cuda")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available but forced in config")
```

---

## Performance Expectations

### Current (MPS Laptop with BiomedCLIP on CPU)

| Operation | Time | Device |
|-----------|------|--------|
| BioBERT embedding (1 doc) | ~50ms | MPS |
| BiomedCLIP image analysis (141 conditions) | ~8s | **CPU** |
| Llama 3 generation (200 tokens) | ~3s | Metal |
| FAISS search (1000 docs) | ~5ms | CPU |

### Expected (CUDA with BiomedCLIP on GPU)

| Operation | Time | Device | Speedup |
|-----------|------|--------|---------|
| BioBERT embedding (1 doc) | ~10-20ms | CUDA | 2-5x |
| BiomedCLIP image analysis (141 conditions) | **~0.5-1s** | **CUDA** | **8-16x** |
| Llama 3 generation (200 tokens) | ~1-2s | CUDA | 1.5-3x |
| FAISS search (1000 docs) | ~1-2ms | GPU | 2-5x |

**Total pipeline**: ~15s → ~3-5s (3-5x faster)

---

## Recommended Testing Workflow

1. **Quick Test** (Google Colab - 5 min setup)
   ```python
   # In Colab notebook
   !git clone <your-repo>
   %cd Codebase
   !pip install -r requirements.txt
   
   # Enable GPU for BiomedCLIP
   !sed -i 's/torch.device("cpu")/torch.device("cuda")/' image_processor.py
   
   # Run test
   !python test_cuda_devices.py
   !python test_biomedclip.py
   ```

2. **Full Test** (Cloud GPU instance - 30 min setup)
   - Launch AWS EC2 g4dn.xlarge (T4 GPU)
   - SSH and run full application
   - Test all workflows

3. **Production Deploy** (Docker + GPU - 1 hour)
   - Build CUDA Docker image
   - Deploy to cloud with GPU
   - Run load tests

---

## Summary

**Your current setup**:
- ✅ BioBERT: MPS (Apple GPU) - optimal for your laptop
- ⚠️ BiomedCLIP: CPU only - could be 8-16x faster on GPU
- ✅ Llama 3: Metal (via Ollama) - optimal for your laptop
- ✅ FAISS: CPU - adequate performance

**To test on CUDA**:
1. **Easiest**: Google Colab (free, 5 min setup)
2. **Best**: Cloud GPU instance (AWS/GCP, $0.50-1/hour)
3. **Flexible**: Docker + remote deployment

**Code changes needed for CUDA**:
- Update `image_processor.py:63` to auto-detect GPU
- Everything else already supports CUDA
