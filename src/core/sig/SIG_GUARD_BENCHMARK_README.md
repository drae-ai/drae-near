# SIG Guard Comprehensive Benchmark

This benchmark evaluates the effectiveness of the Semantic Integrity Guard (`sig_guard.py`) using three carefully selected datasets that test different aspects of semantic similarity detection.

## Overview

The benchmark uses three datasets to comprehensively test the sig_guard:

1. **STS Benchmark (STSb)** - Tests ROC-style discriminative power
2. **PAWS-Wiki/PAWS-QQP** - Tests adversarial pair detection
3. **PARAPHRASUS** - Tests fine-grained paraphrase consistency

## Datasets

### 1. STS Benchmark (STSb)
- **What it is**: 8.6k sentence pairs, each scored 0→5 by crowd workers for graded semantic similarity
- **Source**: Captions and news text (ordinary, human-judgeable language)
- **Why it helps**: Continuous gold scores let you slice pairs into bands that straddle your cosine/Jaccard deltas (e.g., 4.5-5.0 = "should definitely pass", 3.0-3.5 = "should definitely fail")
- **Analysis**: ROC-style checks on whether the guard flips at places that feel right to a human

### 2. PAWS-Wiki / PAWS-QQP
- **What it is**: 108k English pairs where word order is scrambled to create near-identical lexical overlap but sometimes opposite meaning
- **Source**: Human-labeled paraphrase vs. non-paraphrase pairs
- **Why it helps**: Simple bag-of-words or token-distribution metrics often let these "adversarial" pairs slip through
- **Analysis**: If your guard correctly fails the non-paraphrase PAWS pairs—even though Jaccard similarity is high—then you know the cosine/JS components are doing their job

### 3. PARAPHRASUS (COLING 2025)
- **What it is**: Multi-facet benchmark with 10 sub-datasets to probe fine-grained notions of paraphrase, including "borderline" pairs where even humans disagree
- **Source**: Newly annotated datasets covering various paraphrase types
- **Why it helps**: Stress-tests the guard at different strictness levels across genres and paraphrase types
- **Analysis**: Maps fine-grained labels onto "must-pass" vs. "must-fail" categories to test consistency

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy model (if not already installed):
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Quick Start
Run the complete benchmark:
```bash
python run_sig_guard_benchmark.py
```

### Programmatic Usage
```python
from sig_guard_benchmark import SIGGuardBenchmark

# Initialize and run benchmark
benchmark = SIGGuardBenchmark()
success = benchmark.run_full_benchmark()

# Access results
results = benchmark.results
```

## Output

The benchmark generates a comprehensive JSON report (`sig_guard_benchmark_results.json`) containing:

### STS Benchmark Results
- **ROC AUC Score**: Measures discriminative power (higher = better)
- **Pass Rates by Similarity Band**: How often the guard passes for different similarity levels
- **Mean Metrics**: Average cosine distance, JS divergence, and Jaccard similarity

### PAWS Results
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Confusion Matrix**: True/false positives and negatives
- **Metric Analysis**: Failure rates for individual metrics

### PARAPHRASUS Results
- **Consistency Score**: How well the guard behaves across paraphrase types
- **Type Analysis**: Performance breakdown by paraphrase category
- **Sample Results**: Detailed examples for inspection

### Summary
- **Overall Recommendations**: Actionable insights based on performance
- **Benchmark Info**: Settings and configuration used
- **Dataset Summary**: High-level statistics

## Interpreting Results

### Good Performance Indicators
- **STS Benchmark**: ROC AUC > 0.8 indicates strong discriminative power
- **PAWS**: Accuracy > 0.8 shows effective adversarial detection
- **PARAPHRASUS**: Consistency > 0.8 demonstrates reliable behavior across paraphrase types

### Tuning the Guard
Based on benchmark results, you can adjust the `DELTA` values in `sig_guard.py`:

```python
DELTA = {
    "cosine_distance": 0.05,   # Adjust based on STS results
    "js_divergence":   0.10,   # Adjust based on PAWS results
    "jaccard_similarity": 0.05 # Adjust based on PARAPHRASUS results
}
```

## Example Results Interpretation

```json
{
  "summary": {
    "recommendations": [
      "✅ STS Benchmark: Guard shows good discriminative power",
      "✅ PAWS: Guard effectively detects adversarial pairs",
      "⚠️ PARAPHRASUS: Guard shows moderate consistency"
    ]
  }
}
```

This would indicate:
- The guard works well for general similarity detection
- It's effective at catching adversarial examples
- There's room for improvement in fine-grained paraphrase detection

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset Loading Failures**: The benchmark includes fallback data for testing
   - STS Benchmark: Uses GLUE dataset
   - PAWS: Uses HuggingFace datasets
   - PARAPHRASUS: Uses synthetic data (real dataset may not be available yet)

3. **Memory Issues**: For large datasets, the benchmark limits PAWS to 1000 pairs
   - Modify the limit in `load_paws_dataset()` if needed

### Performance Tips

- Use GPU if available (sentence-transformers will automatically detect)
- The benchmark processes pairs in batches to manage memory
- Results are saved incrementally to avoid data loss

## Extending the Benchmark

### Adding New Datasets
1. Create a new loading method in `SIGGuardBenchmark`
2. Add corresponding benchmark method
3. Update the `run_full_benchmark()` method

### Custom Metrics
1. Modify the `compute_metrics()` method
2. Update the guard logic in `sig_guard.py`
3. Adjust the analysis methods accordingly

## Contributing

When adding new datasets or metrics:
1. Follow the existing code structure
2. Include proper error handling
3. Add comprehensive documentation
4. Test with fallback data for robustness

## License

This benchmark is part of the SIG (Semantic Integrity Guarantee) project.
