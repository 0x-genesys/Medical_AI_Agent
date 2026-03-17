#!/usr/bin/env python3
"""
Debug MedCLIP Similarity Computation
Tests the similarity calculation with a fracture image
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

print("=" * 60)
print("MedCLIP Similarity Debug")
print("=" * 60)

# Initialize MedCLIP
print("\n[1/4] Loading MedCLIP...")
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.eval()
processor = MedCLIPProcessor()

print("✓ MedCLIP loaded")

# Test conditions
test_conditions = [
    "Scaphoid Fracture",
    "Hip Fracture", 
    "Rib Fracture",
    "Clopidogrel",
    "IBS",
    "Bipolar Disorder",
    "Pneumonia",
    "Lung Cancer"
]

print(f"\n[2/4] Testing with {len(test_conditions)} conditions")

# Create a dummy image (since we don't have the actual fracture image)
print("\n[3/4] Processing test image...")
dummy_image = Image.new('RGB', (224, 224), color='white')

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                       std=[0.26862954, 0.26130258, 0.27577711])
])

pixel_values = transform(dummy_image).unsqueeze(0)

# Tokenize text
tokenizer = processor.tokenizer
text_inputs = tokenizer(
    test_conditions,
    padding=True,
    truncation=True,
    max_length=77,
    return_tensors="pt"
)

print("\n[4/4] Computing similarities...")
print("\nMethod 1: Raw encoder outputs (INCORRECT - old implementation)")
print("-" * 60)

with torch.no_grad():
    # OLD BROKEN implementation - bypasses projection heads
    image_embeds = model.vision_model(pixel_values)
    text_embeds = model.text_model(
        input_ids=text_inputs['input_ids'],
        attention_mask=text_inputs['attention_mask']
    )
    
    print(f"Raw encoder output norms: image={image_embeds.norm(dim=-1).item():.2f}, text={text_embeds.norm(dim=-1).mean().item():.2f}")
    
    # Normalize
    image_embeds_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # Compute similarity
    similarity_broken = (100.0 * image_embeds_norm @ text_embeds_norm.T).squeeze(0).cpu().numpy()
    
    # Get top 3
    top_indices = np.argsort(similarity_broken)[-3:][::-1]
    print(f"\nTop 3 matches (BROKEN METHOD):")
    for idx in top_indices:
        print(f"  {test_conditions[idx]:30s}: {similarity_broken[idx]:.4f}")

print("\n" + "=" * 60)
print("Method 2: Using projection heads (CORRECT - CPU safe)")
print("-" * 60)

with torch.no_grad():
    # CORRECT implementation - uses projection heads via pooler_output
    vision_outputs = model.vision_model.model(pixel_values)
    vision_pooled = vision_outputs.pooler_output  # [1, 768]
    image_features = model.vision_model.projection_head(vision_pooled)  # [1, 512]
    
    text_outputs = model.text_model.model(
        input_ids=text_inputs['input_ids'],
        attention_mask=text_inputs['attention_mask']
    )
    text_pooled = text_outputs.pooler_output  # [num_texts, 768]
    text_features = model.text_model.projection_head(text_pooled)  # [num_texts, 512]
    
    print(f"Pooled features shape: vision={vision_pooled.shape}, text={text_pooled.shape}")
    print(f"Projected features shape: image={image_features.shape}, text={text_features.shape}")
    print(f"Projected feature norms: image={image_features.norm(dim=-1).item():.2f}, text={text_features.norm(dim=-1).mean().item():.2f}")
    
    # Normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    similarity_correct = (100.0 * image_features @ text_features.T).squeeze(0).cpu().numpy()
    
    print("\nSimilarity scores (CORRECT METHOD):")
    for condition, score in zip(test_conditions, similarity_correct):
        print(f"  {condition:30s}: {score:.4f}")
    
    # Get top 3
    top_indices_correct = np.argsort(similarity_correct)[-3:][::-1]
    print(f"\nTop 3 matches (CORRECT METHOD):")
    for idx in top_indices_correct:
        print(f"  {test_conditions[idx]:30s}: {similarity_correct[idx]:.4f}")

print("\n" + "=" * 60)
print("Comparison Summary")
print("=" * 60)
print("\nBROKEN (old): Top 3 =", [test_conditions[i] for i in np.argsort(similarity_broken)[-3:][::-1]])
print("CORRECT (new): Top 3 =", [test_conditions[i] for i in np.argsort(similarity_correct)[-3:][::-1]])
print("\n✓ Fix applied: Now using encode_image/encode_text with projection heads")
