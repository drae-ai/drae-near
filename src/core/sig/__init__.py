"""
Semantic Integrity Guarantee (SIG) Module

A comprehensive Python library for measuring semantic integrity between two texts
using multiple metrics including cosine distance, Jensen-Shannon divergence, and Jaccard similarity.
"""

from .sig import semantic_integrity_guarantee
from .sig_guard import passes_sig_guard, DELTA

__version__ = "0.1.0"
__author__ = "Drae Team"
__all__ = [
    "semantic_integrity_guarantee",
    "passes_sig_guard",
    "DELTA"
]
