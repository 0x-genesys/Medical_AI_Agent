"""
Test BiomedCLIP loading in isolation to find silent exit cause
"""
import sys
import os
from pathlib import Path

print("=" * 80)
print("BIOMEDCLIP LOADING TEST - Debugging Silent Exit")
print("=" * 80)

# Add Codebase to path
codebase_path = str(Path(__file__).parent)
if codebase_path not in sys.path:
    sys.path.insert(0, codebase_path)

print("\n[Step 1] Checking imports...")
print("-" * 80)

try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"✓ MPS available: {torch.backends.mps.is_available()}")
except Exception as e:
    print(f"❌ PyTorch import failed: {e}")
    sys.exit(1)

try:
    from open_clip import create_model_from_pretrained, get_tokenizer
    print(f"✓ open_clip imported successfully")
except Exception as e:
    print(f"❌ open_clip import failed: {e}")
    print("\nTry: pip install open-clip-torch")
    sys.exit(1)

print("\n[Step 2] Testing BiomedCLIP model loading...")
print("-" * 80)
print("This is the line that causes silent exit in image_processor.py")
print("Model: 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'")
print()

device = torch.device("cpu")
print(f"Using device: {device}")
print()

try:
    print("Attempting to load BiomedCLIP...")
    print("(This may take time to download ~500MB model)")
    print()
    
    clip_model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    
    print("✓ Model loaded successfully!")
    print(f"✓ Model type: {type(clip_model)}")
    print(f"✓ Preprocess type: {type(preprocess)}")
    
    print("\n[Step 3] Moving model to device...")
    clip_model.to(device)
    clip_model.eval()
    print(f"✓ Model moved to {device} and set to eval mode")
    
    print("\n[Step 4] Testing tokenizer...")
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    print(f"✓ Tokenizer loaded: {type(tokenizer)}")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - BiomedCLIP loads successfully!")
    print("=" * 80)
    print("\nIf this succeeds but image_processor.py fails, the issue is")
    print("likely in the context/imports, not the model loading itself.")
    
except Exception as e:
    print(f"\n❌ ERROR during model loading:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print()
    import traceback
    print("Full traceback:")
    traceback.print_exc()
    print()
    print("=" * 80)
    print("POTENTIAL FIXES:")
    print("=" * 80)
    print("1. Check internet connection (model needs to download)")
    print("2. Check disk space (~500MB needed)")
    print("3. Try: pip install --upgrade open-clip-torch")
    print("4. Try: pip install --upgrade huggingface-hub")
    print("5. Clear cache: rm -rf ~/.cache/huggingface/")
    sys.exit(1)

print("\n[Step 5] Cleanup...")
del clip_model
del preprocess
del tokenizer
import gc
gc.collect()
print("✓ Cleanup complete")

print("\n👋 Test complete - exiting normally")
