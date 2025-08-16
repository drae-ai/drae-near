#!/usr/bin/env python3
"""
Simple script to run the semantic integrity guarantee benchmark.
"""

import sys
import os

def main():
    print("🚀 Starting Semantic Integrity Guarantee Benchmark")
    print("=" * 60)

    try:
        # Import and run the benchmark
        from .test_sig import run_benchmark

        success = run_benchmark()

        if success:
            print("\n🎉 Benchmark completed successfully!")
            print("📊 Check 'benchmark_report.json' for detailed results")
        else:
            print("\n❌ Benchmark completed with failures")
            sys.exit(1)

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
