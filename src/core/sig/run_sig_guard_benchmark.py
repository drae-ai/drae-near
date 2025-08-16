#!/usr/bin/env python3
"""
Simple runner script for the SIG Guard benchmark.
"""

import sys
import os

def main():
    print("🚀 Starting SIG Guard Comprehensive Benchmark")
    print("=" * 60)
    print("This benchmark will test the sig_guard using:")
    print("1. STS Benchmark (STSb) - for ROC-style analysis")
    print("2. PAWS-Wiki/PAWS-QQP - for adversarial pair detection")
    print("3. PARAPHRASUS - for fine-grained paraphrase detection")
    print("=" * 60)

    try:
        # Import and run the benchmark
        from .sig_guard_benchmark import SIGGuardBenchmark

        benchmark = SIGGuardBenchmark()
        success = benchmark.run_full_benchmark()

        if success:
            print("\n🎉 SIG Guard benchmark completed successfully!")
            print("📊 Check 'sig_guard_benchmark_results.json' for detailed results")
            print("\n📈 Key metrics to review:")
            print("   - STS Benchmark ROC AUC score")
            print("   - PAWS accuracy for adversarial detection")
            print("   - PARAPHRASUS consistency across paraphrase types")
            print("   - Overall recommendations in the summary")
        else:
            print("\n❌ SIG Guard benchmark completed with failures")
            sys.exit(1)

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
