#!/usr/bin/env python3
"""
Verify what conditions are actually loaded by ImageProcessor
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from image_processor import ImageProcessor
from models import MedicalImage

print("=" * 80)
print("Verifying ImageProcessor Condition Loading")
print("=" * 80)

# Initialize ImageProcessor (same as app does)
print("\n[1/2] Initializing ImageProcessor...")
processor = ImageProcessor()

print(f"\n[2/2] Loaded conditions count: {len(processor.medical_conditions)}")

# Show first 20 conditions
print("\nFirst 20 conditions:")
for i, cond in enumerate(processor.medical_conditions[:20], 1):
    print(f"  {i:2d}. {cond}")

# Check for key indicators
print("\n" + "="*80)
print("Condition Check:")
print("="*80)

# Old format indicators
old_indicators = ['Emphysema', 'Scoliosis', 'Pulmonary Fibrosis']
new_indicators = ['fracture', 'wrist fracture', 'emphysema', 'scoliosis']

print("\nOLD format conditions (capitalized, specific names):")
for ind in old_indicators:
    found = ind in processor.medical_conditions
    status = "✗ FOUND (OLD)" if found else "✓ not found"
    print(f"  {ind:30s}: {status}")

print("\nNEW format conditions (lowercase, general terms):")
for ind in new_indicators:
    found = ind in processor.medical_conditions
    status = "✓ FOUND (NEW)" if found else "✗ not found"
    print(f"  {ind:30s}: {status}")

# Show fracture-related conditions
fracture_conditions = [c for c in processor.medical_conditions if 'fracture' in c.lower()]
print(f"\n" + "="*80)
print(f"Fracture-related conditions ({len(fracture_conditions)} found):")
print("="*80)
for cond in fracture_conditions:
    print(f"  - {cond}")

if not fracture_conditions:
    print("  ⚠ WARNING: No fracture conditions found!")
    print("  This means you're loading the OLD conditions list.")
    print("  Solution: RESTART your application!")

# Test actual similarity computation
print("\n" + "="*80)
print("Testing with Scaphoid Fracture Image")
print("="*80)

image_path = Path(__file__).parent / "examples" / "schapoid_fracture.jpg"
if image_path.exists():
    medical_image = MedicalImage(
        image_path=str(image_path),
        modality="X-ray",
        body_part="wrist"
    )
    
    print(f"\nComputing similarities for: {image_path.name}")
    similarities = processor.compute_image_text_similarity(
        str(image_path),
        processor.medical_conditions
    )
    
    # Get top 5
    import numpy as np
    top_indices = np.argsort(similarities)[-5:][::-1]
    
    print("\nTop 5 MedCLIP matches:")
    for rank, idx in enumerate(top_indices, 1):
        condition = processor.medical_conditions[idx]
        score = similarities[idx]
        print(f"  {rank}. {condition:40s} {score:7.3f}")
    
    # Check if fracture is in top 5
    top_conditions = [processor.medical_conditions[idx] for idx in top_indices]
    has_fracture = any('fracture' in c.lower() for c in top_conditions)
    
    print("\n" + "="*80)
    if has_fracture:
        print("✓ SUCCESS: Fracture detected in top 5!")
        print("Your conditions are properly loaded.")
    else:
        print("✗ FAILURE: No fracture in top 5!")
        print("Your application is using CACHED OLD CONDITIONS.")
        print("\n⚠ SOLUTION: RESTART YOUR APPLICATION!")
else:
    print(f"Image not found: {image_path}")
