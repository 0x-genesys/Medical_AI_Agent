"""
Test to reproduce BiomedCLIP crash in orchestrator initialization sequence
Mimics exact order: TextProcessor (BioBERT+FAISS) → ImageProcessor (BiomedCLIP)
"""
import sys
import os
from pathlib import Path

# CRITICAL: Disable tokenizers parallelism BEFORE any imports
# This prevents multiprocessing conflicts between BioBERT and BiomedCLIP
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'

print("=" * 80)
print("ORCHESTRATOR SEQUENCE TEST - Reproducing BiomedCLIP Crash")
print("=" * 80)
print("Environment: TOKENIZERS_PARALLELISM=false, OMP_NUM_THREADS=1")
print()

# Add Codebase to path
codebase_path = str(Path(__file__).parent)
if codebase_path not in sys.path:
    sys.path.insert(0, codebase_path)

print("\n[Step 1] Initializing TextProcessor (BioBERT + FAISS)...")
print("-" * 80)

try:
    from text_processor import TextProcessor
    
    print("Creating TextProcessor...")
    text_processor = TextProcessor()
    print(f"✓ TextProcessor initialized")
    print(f"  - Device: {text_processor.device}")
    print(f"  - BioBERT loaded: {text_processor.embedder is not None}")
    print(f"  - FAISS indexed: {text_processor.index is not None}")
    print(f"  - Documents: {len(text_processor.documents) if text_processor.documents else 0}")
    
except Exception as e:
    print(f"❌ TextProcessor failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[Step 2] Clearing PyTorch cache and forcing GPU memory release...")
print("-" * 80)

try:
    import torch
    import gc
    
    # Clear MPS cache if BioBERT used it
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("✓ Cleared MPS cache")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ Cleared CUDA cache")
    
    # Force garbage collection
    gc.collect()
    print("✓ Forced garbage collection")
    
    # Check memory
    import psutil
    mem = psutil.virtual_memory()
    print(f"Memory: {mem.available / (1024**3):.2f}GB available / {mem.total / (1024**3):.2f}GB total")
    print(f"Memory used by process: {psutil.Process().memory_info().rss / (1024**3):.2f}GB")
except Exception as e:
    print(f"⚠️  Cache clearing failed: {e}")

print("\n[Step 3] Initializing ImageProcessor (BiomedCLIP)...")
print("-" * 80)
print("⚠️  THIS IS WHERE ORCHESTRATOR CRASHES")
print()

try:
    from image_processor import ImageProcessor
    
    print("Creating ImageProcessor...")
    sys.stdout.flush()
    
    image_processor = ImageProcessor()
    
    print(f"✓ ImageProcessor initialized")
    print(f"  - Device: {image_processor.device}")
    print(f"  - BiomedCLIP loaded: {image_processor.clip_model is not None}")
    print(f"  - Conditions: {len(image_processor.medical_conditions)}")
    
except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ ImageProcessor failed: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[Step 4] Both processors loaded successfully")
print("-" * 80)

try:
    mem = psutil.virtual_memory()
    print(f"Final memory: {mem.available / (1024**3):.2f}GB available")
    print(f"Process memory: {psutil.Process().memory_info().rss / (1024**3):.2f}GB")
except:
    pass

print("\n" + "=" * 80)
print("✅ TEST PASSED - Both processors initialized successfully")
print("=" * 80)
print("\nIf this test PASSES but orchestrator FAILS, the issue is:")
print("1. Orchestrator-specific initialization code")
print("2. MultimodalFusion initialization")
print("3. Session manager conflict")
print("4. Import order in orchestrator")
print()
print("If this test FAILS (crashes), the issue is:")
print("1. TextProcessor + ImageProcessor initialization order conflict")
print("2. BioBERT/FAISS consuming resources that block BiomedCLIP")
print("3. macOS multiprocessing limitations")

# Cleanup
print("\nCleaning up...")
del text_processor
del image_processor
import gc
gc.collect()
print("✓ Cleanup done")

print("\n👋 Exiting normally...")
