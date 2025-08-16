#!/usr/bin/env python3
"""
Simple test script that can be run directly from the sig directory.
This script tests the basic functionality without complex import handling.
"""

import sys
import os

def test_direct_imports():
    """Test importing the modules directly."""
    print("🧪 Testing direct imports...")

    try:
        # Add current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        # Test core imports
        print("  Testing core imports...")
        import sig
        import sig_guard
        print("    ✅ Core modules imported successfully")

        # Test function imports
        from sig import semantic_integrity_guarantee
        from sig_guard import passes_sig_guard, DELTA
        print("    ✅ Core functions imported successfully")

        return True

    except ImportError as e:
        print(f"    ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"    ❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of the main functions."""
    print("\n🧪 Testing basic functionality...")

    try:
        from sig import semantic_integrity_guarantee
        from sig_guard import passes_sig_guard

        # Test with simple texts
        text1 = "Hello world"
        text2 = "Hello world"

        print(f"    Testing with texts: '{text1}' vs '{text2}'")
        result = semantic_integrity_guarantee(text1, text2)
        print(f"    ✅ semantic_integrity_guarantee: {result}")

        # Test sig guard
        baseline = {'cosine_distance': 0.1, 'js_divergence': 0.1, 'jaccard_similarity': 0.9}
        candidate = {'cosine_distance': 0.15, 'js_divergence': 0.12, 'jaccard_similarity': 0.85}

        print(f"    Testing sig guard with baseline: {baseline}")
        print(f"    Testing sig guard with candidate: {candidate}")
        passes = passes_sig_guard(baseline, candidate)
        print(f"    ✅ passes_sig_guard: {passes}")

        print("\n🎉 Basic functionality test successful!")
        return True

    except Exception as e:
        print(f"    ❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependencies():
    """Test if all required dependencies are available."""
    print("\n🧪 Testing dependencies...")

    required_packages = [
        'numpy', 'spacy', 'sentence_transformers',
        'sklearn', 'scipy'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"    ✅ {package} available")
        except ImportError:
            print(f"    ❌ {package} missing")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n🎉 All required dependencies are available!")
        return True

if __name__ == "__main__":
    print("🚀 SIG Module Simple Test")
    print("=" * 40)

    # Test dependencies first
    deps_ok = test_dependencies()
    if not deps_ok:
        print("\n❌ Dependency test failed. Please install missing packages first.")
        sys.exit(1)

    # Test imports
    import_ok = test_direct_imports()
    if not import_ok:
        print("\n❌ Import test failed.")
        sys.exit(1)

    # Test functionality
    func_ok = test_basic_functionality()
    if not func_ok:
        print("\n❌ Functionality test failed.")
        sys.exit(1)

    print("\n🎉 All tests passed! The SIG module is working correctly.")
    print("\nYou can now use the SIG module in your code:")
    print("  from sig import semantic_integrity_guarantee")
    print("  from sig_guard import passes_sig_guard")
