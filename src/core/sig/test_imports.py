#!/usr/bin/env python3
"""
Test script to verify that all imports are working correctly.
Run this script to check if the dependency and import issues are resolved.
"""

def test_imports():
    """Test all the main imports from the sig module."""
    print("🧪 Testing imports...")

    try:
        # Test core imports
        print("  Testing core imports...")
        try:
            # Try relative imports first (when run as module)
            from .sig import semantic_integrity_guarantee
            from .sig_guard import passes_sig_guard, DELTA
            print("    ✅ Core imports successful (relative)")
        except ImportError:
            # Fall back to absolute imports (when run directly)
            from sig import semantic_integrity_guarantee
            from sig_guard import passes_sig_guard, DELTA
            print("    ✅ Core imports successful (absolute)")

        # Test benchmark imports
        print("  Testing benchmark imports...")
        try:
            from .sig_guard_benchmark import SIGGuardBenchmark
            print("    ✅ Benchmark imports successful (relative)")
        except ImportError:
            from sig_guard_benchmark import SIGGuardBenchmark
            print("    ✅ Benchmark imports successful (absolute)")

        # Test auto-tune imports
        print("  Testing auto-tune imports...")
        try:
            from .sig_guard_auto_tune import get_benchmark_score
            print("    ✅ Auto-tune imports successful (relative)")
        except ImportError:
            from sig_guard_auto_tune import get_benchmark_score
            print("    ✅ Auto-tune imports successful (absolute)")

        print("\n🎉 All imports successful!")
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
        # Try to import functions
        try:
            from .sig import semantic_integrity_guarantee
            from .sig_guard import passes_sig_guard
        except ImportError:
            from sig import semantic_integrity_guarantee
            from sig_guard import passes_sig_guard

        # Test with simple texts
        text1 = "Hello world"
        text2 = "Hello world"

        result = semantic_integrity_guarantee(text1, text2)
        print(f"    ✅ semantic_integrity_guarantee: {result}")

        # Test sig guard
        baseline = {'cosine_distance': 0.1, 'js_divergence': 0.1, 'jaccard_similarity': 0.9}
        candidate = {'cosine_distance': 0.15, 'js_divergence': 0.12, 'jaccard_similarity': 0.85}

        passes = passes_sig_guard(baseline, candidate)
        print(f"    ✅ passes_sig_guard: {passes}")

        print("\n🎉 Basic functionality test successful!")
        return True

    except Exception as e:
        print(f"    ❌ Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 SIG Module Import and Functionality Test")
    print("=" * 50)

    import_success = test_imports()
    if import_success:
        func_success = test_basic_functionality()
        if func_success:
            print("\n🎉 All tests passed! The SIG module is working correctly.")
        else:
            print("\n❌ Functionality tests failed.")
            exit(1)
    else:
        print("\n❌ Import tests failed.")
        exit(1)
