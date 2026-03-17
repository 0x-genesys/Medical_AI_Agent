#!/usr/bin/env python3
"""
Post-Install OpenMP Fix
Run this after pip install requirements.txt to prevent OpenMP conflicts

This script consolidates duplicate libomp.dylib files from different packages
(PyTorch, FAISS, scikit-learn) to use a single shared version.

Usage:
    python fix_openmp.py
"""
import os
import sys
from pathlib import Path
import shutil

def find_site_packages():
    """Find the site-packages directory in current environment"""
    for path in sys.path:
        if 'site-packages' in path and os.path.isdir(path):
            return Path(path)
    raise RuntimeError("Could not find site-packages directory")

def consolidate_openmp():
    """Consolidate duplicate OpenMP libraries to prevent crashes"""
    print("=" * 60)
    print("OpenMP Library Consolidation")
    print("=" * 60)
    
    site_packages = find_site_packages()
    print(f"\nSite-packages: {site_packages}")
    
    # Locate OpenMP libraries
    torch_omp = site_packages / "torch" / "lib" / "libomp.dylib"
    sklearn_omp = site_packages / "sklearn" / ".dylibs" / "libomp.dylib"
    faiss_omp = site_packages / "faiss" / ".dylibs" / "libomp.dylib"
    
    # Check which libraries exist
    libs_found = []
    if torch_omp.exists():
        libs_found.append(f"PyTorch: {torch_omp}")
    if sklearn_omp.exists():
        libs_found.append(f"scikit-learn: {sklearn_omp}")
    if faiss_omp.exists():
        libs_found.append(f"FAISS: {faiss_omp}")
    
    if len(libs_found) <= 1:
        print("\n✓ No duplicate OpenMP libraries found. Nothing to do.")
        return
    
    print(f"\nFound {len(libs_found)} OpenMP libraries:")
    for lib in libs_found:
        print(f"  • {lib}")
    
    if not torch_omp.exists():
        print("\n⚠ PyTorch OpenMP library not found. Cannot consolidate.")
        print("  Make sure PyTorch is installed: pip install torch")
        return
    
    print(f"\n→ Will consolidate all to use PyTorch's version")
    
    # Consolidate scikit-learn
    if sklearn_omp.exists():
        if sklearn_omp.is_symlink():
            print(f"\n  ✓ sklearn already symlinked")
        else:
            backup = sklearn_omp.with_suffix('.dylib.backup')
            print(f"\n  • Backing up sklearn OpenMP to {backup.name}")
            shutil.move(str(sklearn_omp), str(backup))
            
            print(f"  • Creating symlink: sklearn → torch")
            rel_path = os.path.relpath(torch_omp, sklearn_omp.parent)
            sklearn_omp.symlink_to(rel_path)
            print(f"  ✓ sklearn consolidated")
    
    # Consolidate FAISS
    if faiss_omp.exists():
        if faiss_omp.is_symlink():
            print(f"\n  ✓ FAISS already symlinked")
        else:
            backup = faiss_omp.with_suffix('.dylib.backup')
            print(f"\n  • Backing up FAISS OpenMP to {backup.name}")
            shutil.move(str(faiss_omp), str(backup))
            
            print(f"  • Creating symlink: FAISS → torch")
            rel_path = os.path.relpath(torch_omp, faiss_omp.parent)
            faiss_omp.symlink_to(rel_path)
            print(f"  ✓ FAISS consolidated")
    
    print("\n" + "=" * 60)
    print("✓ OpenMP consolidation complete!")
    print("=" * 60)
    print("\nAll packages now use a single shared OpenMP library.")
    print("This prevents SIGSEGV crashes during parallel operations.")

def main():
    try:
        consolidate_openmp()
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
