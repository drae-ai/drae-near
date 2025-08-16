# sig_guard.py
import math
from typing import Dict, Optional

# Tunable tolerances (can be made env-configurable)
DELTA = {
    "cosine_distance": 0.1,   # max allowed increase
    "js_divergence":   0.1,   # max allowed increase
    "jaccard_similarity": 0.05 # max allowed decrease
}

def _is_nan(x: Optional[float]) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))

def passes_sig_guard(baseline: Dict[str, float],
                     candidate: Dict[str, float],
                     delta: Dict[str, float] = DELTA) -> bool:
    """
    Return True iff *every* present metric stays within the allowed delta
    relative to the recorded baseline.

    • For 'distance' / 'divergence' metrics we allow only *increases* ≤ Δ.
    • For 'similarity' metrics we allow only *decreases* ≤ Δ.
    • Missing or NaN metrics are ignored (treated as non-gating).
    """
    for k, tol in delta.items():
        if _is_nan(baseline.get(k)) or _is_nan(candidate.get(k)):
            continue  # nothing to compare

        if k in ("cosine_distance", "js_divergence"):
            if candidate[k] > baseline[k] + tol:
                return False  # degraded
        elif k == "jaccard_similarity":
            if candidate[k] < baseline[k] - tol:
                return False  # degraded
    return True
