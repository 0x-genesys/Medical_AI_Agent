#!/usr/bin/env python3
"""
Test script to verify device availability and model loading
Tests current configuration: BiomedCLIP on CPU, BioBERT auto-detects GPU
"""

import sys
import os

# Set OpenMP environment before imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '4'

print("="*70)
print("Device Capability & Model Testing")
print("="*70)
print()

# Test 1: Check PyTorch and device availability
print("[1/6] PyTorch Device Availability:")
print("-" * 70)
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
    else:
        print("  ❌ CUDA not available")
    
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print("  ✓ Apple Silicon GPU detected")
    
    print("  ✓ PyTorch imported successfully")
except Exception as e:
    print(f"  ❌ PyTorch test failed: {e}")
    sys.exit(1)

print()

# Test 2: BioBERT (Auto-detects GPU)
print("[2/6] BioBERT / ClinicalBERT:")
print("-" * 70)
try:
    from sentence_transformers import SentenceTransformer
    
    # Auto-detect device (same as text_processor.py)
    if torch.cuda.is_available():
        biobert_device = "cuda"
        print(f"  Device: CUDA GPU ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        biobert_device = "mps"
        print(f"  Device: MPS (Apple Silicon)")
    else:
        biobert_device = "cpu"
        print(f"  Device: CPU")
    
    print(f"  Loading Bio_ClinicalBERT...")
    embedder = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT', device=biobert_device)
    
    # Test embedding
    test_text = "Patient presents with acute chest pain and dyspnea"
    embedding = embedder.encode(test_text)
    
    print(f"  ✓ BioBERT loaded successfully")
    print(f"  Embedding dimension: {embedding.shape[0]}")
    print(f"  Test encoding: SUCCESS")
    
except Exception as e:
    print(f"  ❌ BioBERT test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 3: BiomedCLIP (CPU-only as per current config)
print("[3/6] BiomedCLIP (Current Config: CPU-only):")
print("-" * 70)
try:
    from open_clip import create_model_from_pretrained, get_tokenizer
    from PIL import Image
    import numpy as np
    
    # CURRENT CONFIGURATION: CPU only (matches image_processor.py)
    biomedclip_device = torch.device("cpu")
    print(f"  Device: CPU (as configured in image_processor.py)")
    print(f"  Note: This is intentional - keeping CPU for consistency")
    
    print(f"  Loading BiomedCLIP...")
    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    
    model.to(biomedclip_device)
    model.eval()
    
    print(f"  ✓ BiomedCLIP loaded successfully")
    
    # Test with dummy image if available
    try:
        # Create a dummy image for testing
        dummy_image = Image.new('RGB', (224, 224), color='gray')
        image_tensor = preprocess(dummy_image).unsqueeze(0).to(biomedclip_device)
        
        # Test text encoding
        test_conditions = ["pneumonia", "fracture", "normal"]
        texts = tokenizer(test_conditions, context_length=256).to(biomedclip_device)
        
        # Compute features
        with torch.no_grad():
            image_features, text_features, logit_scale = model(image_tensor, texts)
            similarities = (logit_scale * image_features @ text_features.t()).squeeze(0)
        
        print(f"  Image encoding: SUCCESS")
        print(f"  Text encoding: SUCCESS (3 conditions)")
        print(f"  Similarity computation: SUCCESS")
        
    except Exception as test_error:
        print(f"  ⚠️  Inference test skipped: {test_error}")
    
except Exception as e:
    print(f"  ❌ BiomedCLIP test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 4: FAISS
print("[4/6] FAISS Vector Search:")
print("-" * 70)
try:
    import faiss
    import numpy as np
    
    print(f"  FAISS version: {faiss.__version__}")
    
    # Check if GPU version
    has_gpu_support = False
    try:
        gpu_count = faiss.get_num_gpus()
        has_gpu_support = True
        print(f"  GPU support: YES ({gpu_count} GPUs available)")
    except:
        print(f"  GPU support: NO (CPU-only build)")
    
    # Test index creation (matches text_processor.py)
    dimension = 768  # BioBERT embedding dimension
    index = faiss.IndexFlatL2(dimension)
    
    # Add some dummy vectors
    dummy_vectors = np.random.randn(100, dimension).astype('float32')
    index.add(dummy_vectors)
    
    # Test search
    query = np.random.randn(1, dimension).astype('float32')
    distances, indices = index.search(query, 5)
    
    print(f"  ✓ FAISS index created (dimension: {dimension})")
    print(f"  ✓ Vector addition: SUCCESS (100 vectors)")
    print(f"  ✓ Similarity search: SUCCESS (top-5)")
    
except Exception as e:
    print(f"  ❌ FAISS test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 5: Ollama
print("[5/6] Ollama LLM:")
print("-" * 70)
try:
    import subprocess
    
    # Check if Ollama is installed
    ollama_check = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
    if ollama_check.returncode == 0:
        print(f"  Ollama path: {ollama_check.stdout.strip()}")
        
        # List models
        list_result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if list_result.returncode == 0:
            print(f"  ✓ Ollama is operational")
            print(f"  Installed models:")
            for line in list_result.stdout.strip().split('\n'):
                if line and 'NAME' not in line:
                    print(f"    - {line.split()[0]}")
        else:
            print(f"  ⚠️  Ollama installed but not responding")
    else:
        print(f"  ❌ Ollama not found in PATH")
        print(f"  Install: curl -fsSL https://ollama.com/install.sh | sh")
    
    # Test LangChain integration
    try:
        from langchain_ollama import OllamaLLM
        llm = OllamaLLM(model='llama3.2', temperature=0.7)
        print(f"  ✓ LangChain integration: SUCCESS")
    except Exception as e:
        print(f"  ⚠️  LangChain integration test skipped: {e}")
        
except Exception as e:
    print(f"  ❌ Ollama test failed: {e}")

print()

# Test 6: Import all processors
print("[6/6] Application Modules:")
print("-" * 70)
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    print("  Importing text_processor...")
    from text_processor import TextProcessor
    print("  ✓ TextProcessor imported")
    
    print("  Importing image_processor...")
    from image_processor import ImageProcessor
    print("  ✓ ImageProcessor imported")
    
    print("  Importing multimodal_fusion...")
    from multimodal_fusion import MultimodalFusion
    print("  ✓ MultimodalFusion imported")
    
    print("  Importing models...")
    from models import MedicalText, MedicalImage, QueryRequest
    print("  ✓ Models imported")
    
    print()
    print("  ✓ All application modules imported successfully")
    
except Exception as e:
    print(f"  ❌ Module import failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*70)
print("Testing Complete")
print("="*70)
print()

# Summary
print("SUMMARY:")
print("-" * 70)
print("Current Configuration (as per codebase):")
print("  • BioBERT: Auto-detect GPU (CUDA → MPS → CPU)")
print("  • BiomedCLIP: CPU only (hardcoded)")
print("  • Llama 3: Managed by Ollama (auto GPU)")
print("  • FAISS: CPU (default)")
print()
print("To enable BiomedCLIP GPU:")
print("  Edit image_processor.py line 63:")
print("  Change: self.device = torch.device('cpu')")
print("  To: Auto-detect logic (see MODEL_DEVICE_USAGE.md)")
print()
print("="*70)
