#!/usr/bin/env python3
"""
Simple installation script for the SIG module.
Run this script to install the module in development mode.
"""

import subprocess
import sys
import os

def main():
    """Install the SIG module in development mode."""
    print("🚀 Installing SIG module in development mode...")

    # Check if we're in the right directory
    if not os.path.exists("setup.py"):
        print("❌ Error: setup.py not found. Please run this script from the sig directory.")
        sys.exit(1)

    # Check if dependencies are installed
    print("📦 Checking dependencies...")
    required_packages = ['numpy', 'spacy', 'sentence_transformers', 'sklearn', 'scipy']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"    ✅ {package} available")
        except ImportError:
            print(f"    ❌ {package} missing")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Installing dependencies first...")

        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], check=True)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("Please install them manually:")
            print(f"pip install {' '.join(missing_packages)}")
            sys.exit(1)

    # Install the module in development mode
    print("\n🔧 Installing SIG module...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], check=True)

        print("✅ SIG module installed successfully!")
        print("\nYou can now import it like this:")
        print("  from src.core.sig import semantic_integrity_guarantee")
        print("  from src.core.sig import passes_sig_guard")
        print("\nOr test it with:")
        print("  python test_simple.py")

    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print("\nTrying alternative installation method...")

        try:
            # Try installing without setup.py
            subprocess.run([
                sys.executable, "-m", "pip", "install", "numpy", "spacy", "sentence-transformers", "scikit-learn", "scipy"
            ], check=True)
            print("✅ Dependencies installed successfully")
            print("\nNote: Module not installed as package, but you can import directly:")
            print("  from sig import semantic_integrity_guarantee")
            print("  from sig_guard import passes_sig_guard")
        except subprocess.CalledProcessError as e2:
            print(f"❌ Alternative installation also failed: {e2}")
            sys.exit(1)

if __name__ == "__main__":
    main()
