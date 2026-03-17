# MedCLIP Investigation Summary

## Problem
MedCLIP pretrained weights are giving incorrect results for fracture detection:
- **Expected**: "fracture" in top 3 for wrist fracture X-ray
- **Actual**: "soft tissue mass", "pleural effusion", "renal cell carcinoma" (fracture ranks #133/141)

## Findings

### 1. Weight File Comparison
**Location**: `Capstone_Project-CS[ID]/pretrained/`
- `medclip/pytorch_model.bin` (522MB) - position_ids removed
- `medclip-vit/pytorch_model.bin` (522MB) - original download

**Result**: Files are **identical** (only difference was position_ids key which causes compatibility error)

### 2. MedCLIP Library Issue
The `medclip` Python library uses weights from:
- URL: `https://storage.googleapis.com/pytrial/medclip-vit-pretrained.zip`
- Source: Research project weights, not official release

**Problem**: These weights give inverted similarity scores:
- Random initialization → fracture ranks #2 ✓
- Pretrained weights → fracture ranks #133 ✗

### 3. HuggingFace Investigation

#### Option A: Old MedCLIP (Current)
- Library: `pip install medclip`
- Weights: PyTrial Google Storage
- Status: **Not working correctly**

#### Option B: Microsoft BiomedCLIP (Recommended)
- **Model**: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- **HuggingFace**: https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- **Library**: `pip install open_clip_torch`
- **Pretrained on**: PMC-15M (15M medical image-caption pairs from PubMed Central)
- **Status**: Official Microsoft release, actively maintained

**BiomedCLIP Usage**:
```python
from open_clip import create_model_from_pretrained, get_tokenizer

model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
```

## Current Status

### ✅ Completed
1. Moved pretrained weights to project root: `Capstone_Project-CS[ID]/pretrained/`
2. Updated `image_processor.py` to load from project root
3. Updated `main.py` download logic to use project root
4. Added `.gitignore` to exclude weights from git

### 📁 Project Structure
```
Capstone_Project-CS[ID]/
├── pretrained/           # ← Project-level (new)
│   ├── medclip/
│   │   └── pytorch_model.bin
│   ├── medclip-vit/
│   │   └── pytorch_model.bin
│   └── .gitignore
└── Codebase/
    ├── main.py           # Updated paths
    ├── image_processor.py # Updated paths
    └── ...
```

## Recommendations

### Short-term Fix (Test Current Setup)
1. Clear Python cache: `rm -rf __pycache__`
2. Restart application
3. Test with scaphoid fracture image

### Long-term Solution (Switch to BiomedCLIP)
1. Install: `pip install open_clip_torch==2.23.0`
2. Replace MedCLIP with BiomedCLIP in `image_processor.py`
3. Benefits:
   - Official Microsoft model
   - Better performance on medical images
   - Active maintenance
   - Larger training dataset (PMC-15M)

## Next Steps
1. **Test current setup** with updated paths
2. **If still failing**: Switch to BiomedCLIP
3. **Document** which model works for production
