#!/usr/bin/env python3
"""
Test Microsoft BiomedCLIP as replacement for MedCLIP
Verify it works on CPU and gives correct fracture detection
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'

import warnings
warnings.filterwarnings('ignore')

import sys
import torch
from pathlib import Path
from PIL import Image

print("=" * 80)
print("Testing Microsoft BiomedCLIP")
print("=" * 80)

# Check if open_clip_torch is installed
try:
    from open_clip import create_model_from_pretrained, get_tokenizer
    print("✓ open_clip_torch installed")
except ImportError:
    print("\n✗ open_clip_torch not installed")
    print("\nInstall with:")
    print("  pip install open_clip_torch==2.23.0")
    sys.exit(1)

# Load medical conditions
conditions_file = Path(__file__).parent / "data" / "imaging_conditions.txt"
conditions = []
with open(conditions_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            conditions.append(line)

print(f"\n[1/4] Loaded {len(conditions)} medical conditions")

# Load scaphoid fracture image
image_path = Path(__file__).parent / "examples" / "schapoid_fracture.jpg"
print(f"\n[2/4] Loading image: {image_path.name}")

if not image_path.exists():
    print(f"✗ Image not found: {image_path}")
    sys.exit(1)

# Load BiomedCLIP model
print("\n[3/4] Loading BiomedCLIP from HuggingFace (will download on first run)...")
print("      Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
print("      This may take a few minutes on first run...")

device = torch.device('cpu')  # Force CPU
print(f"      Device: {device}")

try:
    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    
    model.to(device)
    model.eval()
    
    print("✓ BiomedCLIP loaded successfully")
except Exception as e:
    print(f"\n✗ Failed to load BiomedCLIP: {e}")
    sys.exit(1)

# Compute similarities
print("\n[4/4] Computing similarities for scaphoid fracture image...")

# Preprocess image
image = Image.open(image_path).convert('RGB')
image_tensor = preprocess(image).unsqueeze(0).to(device)

# Tokenize conditions
context_length = 256
texts = tokenizer(conditions, context_length=context_length).to(device)

# Compute features and similarity
with torch.no_grad():
    image_features, text_features, logit_scale = model(image_tensor, texts)
    
    # Compute similarity scores
    # BiomedCLIP returns logit_scale as a learned parameter
    logits = (logit_scale * image_features @ text_features.t()).detach()
    similarities = logits.squeeze(0).cpu().numpy()

print(f"✓ Computed similarities")

# Get top 10
import numpy as np
top_indices = np.argsort(similarities)[-10:][::-1]

print("\n" + "="*80)
print("Top 10 BiomedCLIP Matches:")
print("="*80)
for rank, idx in enumerate(top_indices, 1):
    condition = conditions[idx]
    score = similarities[idx]
    marker = " ← FRACTURE!" if 'fracture' in condition.lower() else ""
    print(f"{rank:2d}. {condition:40s} {score:8.4f}{marker}")

# Check for fracture in top 3
top_3_conditions = [conditions[idx] for idx in top_indices[:3]]
has_fracture_top3 = any('fracture' in c.lower() for c in top_3_conditions)

# Check for fracture in top 10
has_fracture_top10 = any('fracture' in c.lower() for c in [conditions[idx] for idx in top_indices])

# Show all fracture conditions
fracture_conditions = [(idx, cond, similarities[idx]) for idx, cond in enumerate(conditions) if 'fracture' in cond.lower()]
fracture_conditions.sort(key=lambda x: x[2], reverse=True)

print("\n" + "="*80)
print(f"All Fracture-Related Conditions ({len(fracture_conditions)} total):")
print("="*80)
for idx, cond, score in fracture_conditions[:10]:
    rank = np.sum(similarities > score) + 1
    print(f"  {cond:40s} {score:8.4f}  (rank {rank})")

# Final verdict
print("\n" + "="*80)
print("VERDICT:")
print("="*80)

if has_fracture_top3:
    print("✓ SUCCESS: Fracture detected in TOP 3!")
    print("\nBiomedCLIP is working correctly for fracture detection.")
    print("Ready to integrate into image_processor.py")
elif has_fracture_top10:
    print("⚠ PARTIAL: Fracture in top 10 but not top 3")
    print("\nBiomedCLIP works but may need prompt tuning.")
else:
    print("✗ FAILURE: No fracture in top 10")
    print("\nBiomedCLIP may not be suitable for this use case.")

print("\n" + "="*80)
print("Comparison with old MedCLIP:")
print("  MedCLIP (pretrained): fracture ranked #133/141 ✗")
print(f"  BiomedCLIP:          fracture ranked #{np.sum(similarities > fracture_conditions[0][2]) + 1}/141")
print("="*80)
