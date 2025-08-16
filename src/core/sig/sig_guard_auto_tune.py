#!/usr/bin/env python3
"""
Auto-tuner for SIG Guard DELTA parameters.
Performs grid search over DELTA values to maximize a composite benchmark score.
"""
import itertools
import importlib
import json
import time
from copy import deepcopy
from .sig_guard_benchmark import SIGGuardBenchmark
try:
    from . import sig_guard
except ImportError:
    import sig_guard

# Search space for DELTA values
COSINE_RANGE = [0.05, 0.10, 0.15, 0.20]
JS_RANGE = [0.10, 0.15, 0.20, 0.25, 0.30]
JACCARD_RANGE = [0.05, 0.10, 0.15, 0.20]

# Composite score weights
WEIGHTS = {
    'sts_auc': 0.4,         # STS ROC AUC (higher is better)
    'paws_acc': 0.4,        # PAWS accuracy (higher is better)
    'paraphrasus_cons': 0.2 # PARAPHRASUS consistency (higher is better)
}

def get_benchmark_score(results):
    """Compute a composite score from benchmark results."""
    try:
        sts_auc = results['sts_benchmark'].get('roc_auc', 0)
        paws_acc = results['paws']['performance_metrics'].get('accuracy', 0)
        paraphrasus_cons = results['paraphrasus'].get('consistency_score', 0)
        score = (
            WEIGHTS['sts_auc'] * sts_auc +
            WEIGHTS['paws_acc'] * paws_acc +
            WEIGHTS['paraphrasus_cons'] * paraphrasus_cons
        )
        return score, sts_auc, paws_acc, paraphrasus_cons
    except Exception as e:
        print(f"[WARN] Could not compute score: {e}")
        return 0, 0, 0, 0

def main():
    print("\n🚀 Starting SIG Guard Auto-Tuner\n" + "="*60)
    best_score = -1
    best_delta = None
    best_results = None
    search_space = list(itertools.product(COSINE_RANGE, JS_RANGE, JACCARD_RANGE))
    total = len(search_space)
    start_time = time.time()

    for idx, (cosine, js, jaccard) in enumerate(search_space):
        print(f"\n[{idx+1}/{total}] Testing DELTA: cosine={cosine}, js={js}, jaccard={jaccard}")
        # Set DELTA in sig_guard
        sig_guard.DELTA['cosine_distance'] = cosine
        sig_guard.DELTA['js_divergence'] = js
        sig_guard.DELTA['jaccard_similarity'] = jaccard

        # Run benchmark with this DELTA
        benchmark = SIGGuardBenchmark()
        success = benchmark.run_full_benchmark()
        if not success:
            print("[WARN] Benchmark failed for this DELTA.")
            continue
        score, sts_auc, paws_acc, paraphrasus_cons = get_benchmark_score(benchmark.results)
        print(f"  Composite score: {score:.4f} | STS AUC: {sts_auc:.3f} | PAWS acc: {paws_acc:.3f} | PARAPHRASUS cons: {paraphrasus_cons:.3f}")
        if score > best_score:
            best_score = score
            best_delta = {'cosine_distance': cosine, 'js_divergence': js, 'jaccard_similarity': jaccard}
            best_results = deepcopy(benchmark.results)
    elapsed = time.time() - start_time
    print("\n=== Auto-Tuning Complete ===")
    print(f"Best DELTA: {best_delta}")
    print(f"Best composite score: {best_score:.4f}")
    print(f"  STS ROC AUC: {best_results['sts_benchmark'].get('roc_auc', 0):.3f}")
    print(f"  PAWS accuracy: {best_results['paws']['performance_metrics'].get('accuracy', 0):.3f}")
    print(f"  PARAPHRASUS consistency: {best_results['paraphrasus'].get('consistency_score', 0):.3f}")
    # Save best results
    with open('sig_guard_auto_tune_best.json', 'w') as f:
        json.dump({'best_delta': best_delta, 'best_score': best_score, 'results': best_results}, f, indent=2)
    print(f"\nResults saved to sig_guard_auto_tune_best.json")
    print(f"Total tuning time: {elapsed/60:.1f} min")

if __name__ == '__main__':
    main()
