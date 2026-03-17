# Test Suite

This directory contains test scripts for the Multimodal Medical Assistant.

## Test Files

### `test_biomedclip_loading.py`
Tests BiomedCLIP model loading in isolation to debug crashes and initialization issues.

**Usage:**
```bash
cd /Users/k0a05wi/Downloads/Capstone_Project-CS[ID]/Codebase
python tests/test_biomedclip_loading.py
```

### `test_orchestrator_sequence.py`
Tests the exact initialization sequence of the orchestrator (TextProcessor → ImageProcessor) to reproduce and debug model loading conflicts.

**Usage:**
```bash
cd /Users/k0a05wi/Downloads/Capstone_Project-CS[ID]/Codebase
python tests/test_orchestrator_sequence.py
```

### `test_biomedclip.py`
Tests BiomedCLIP inference and functionality.

**Usage:**
```bash
cd /Users/k0a05wi/Downloads/Capstone_Project-CS[ID]/Codebase
python tests/test_biomedclip.py
```

### `test_cuda_devices.py`
Tests CUDA/MPS device detection and availability.

**Usage:**
```bash
cd /Users/k0a05wi/Downloads/Capstone_Project-CS[ID]/Codebase
python tests/test_cuda_devices.py
```

## Important Notes

**Environment Variables:**
All tests should be run from the Codebase directory (parent of tests/) to ensure proper imports and environment setup.

**Critical Fix:**
The `OMP_NUM_THREADS=1` environment variable must be set in `image_processor.py` to prevent OpenMP thread pool conflicts between BioBERT and BiomedCLIP.
