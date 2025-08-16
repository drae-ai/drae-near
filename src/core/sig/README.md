# Semantic Integrity Guarantee (SIG)

A comprehensive Python library for measuring semantic integrity between two texts using multiple metrics.

## Features

- **Cosine Distance**: Measures semantic similarity using sentence embeddings
- **Jensen-Shannon Divergence**: Compares token/ngram distributions with smoothing
- **Jaccard Similarity**: Measures lexical overlap between token sets
- **Comprehensive Benchmarking**: Full test suite with performance metrics

## Installation

### Option 1: Install from the project root (Recommended)
```bash
# From the drae-near project root
pip install -r requirements.txt
cd src/core/sig
python install.py
```

### Option 2: Install dependencies only
```bash
cd src/core/sig
pip install -r requirements.txt
```

### Option 3: Install as a package
```bash
cd src/core/sig
pip install -e .
```

### Download spaCy model (optional, for better tokenization):
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

```python
# If installed as a package
from src.core.sig import semantic_integrity_guarantee

# Or if running from the sig directory
from sig import semantic_integrity_guarantee

# Compare two texts
result = semantic_integrity_guarantee(
    "The weather is beautiful today.",
    "Today's weather is lovely."
)

print(f"Cosine Distance: {result['cosine_distance']:.4f}")
print(f"JS Divergence: {result['js_divergence']:.4f}")
print(f"Jaccard Similarity: {result['jaccard_similarity']:.4f}")
```

### Advanced Usage

```python
from sentence_transformers import SentenceTransformer
import spacy

# Pre-load models for better performance
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load('en_core_web_sm')

result = semantic_integrity_guarantee(
    text1="Machine learning algorithms require large datasets.",
    text2="Deep learning models need extensive data.",
    embedding_model=embedding_model,
    spacy_nlp=nlp,
    ngram_range=(1, 2),  # Use bigrams
    smoothing=1e-4       # Adjust smoothing
)
```

## Metrics Explained

### Cosine Distance
- **Range**: 0 (identical) to 1 (maximally different)
- **Interpretation**: Lower values indicate more similar semantic meaning
- **Based on**: Sentence embeddings from transformer models

### Jensen-Shannon Divergence
- **Range**: 0 (identical) to 1 (maximally different)
- **Interpretation**: Lower values indicate more similar token distributions
- **Based on**: Smoothed n-gram frequency distributions

### Jaccard Similarity
- **Range**: 0 (no overlap) to 1 (identical)
- **Interpretation**: Higher values indicate more lexical overlap
- **Based on**: Set intersection of tokens

## Benchmarking

Run the comprehensive benchmark suite:

```bash
python run_benchmark.py
```

Or run individual tests:

```bash
python -m unittest test_sig.SemanticIntegrityBenchmark.test_basic_functionality
```

### Benchmark Features

- **Performance Testing**: Execution time measurements
- **Consistency Testing**: Multiple runs to ensure reproducibility
- **Edge Case Testing**: Empty texts, special characters, etc.
- **Parameter Variation**: Different n-gram ranges and smoothing values
- **Metric Validation**: Ensures proper value ranges and relationships

### Benchmark Output

The benchmark generates:
- Console output with test results
- `benchmark_report.json` with detailed statistics
- Performance metrics by test case type

## API Reference

### `semantic_integrity_guarantee()`

**Parameters:**
- `text1` (str): Source/original text
- `text2` (str): Generated/transformed text
- `embedding_model` (SentenceTransformer, optional): Pre-loaded embedding model
- `embedding_model_name` (str): Model name if embedding_model not provided (default: 'all-MiniLM-L6-v2')
- `ngram_range` (tuple): n-gram range for distribution metrics (default: (1, 1))
- `smoothing` (float): Additive smoothing for probability distributions (default: 1e-6)
- `spacy_nlp` (spacy.language.Language, optional): Pre-loaded spaCy tokenizer

**Returns:**
```python
{
    'cosine_distance': float,      # 0-1, lower = more similar
    'js_divergence': float,        # 0-1, lower = more similar
    'jaccard_similarity': float    # 0-1, higher = more similar
}
```

## Performance Notes

- First run may be slower due to model loading
- Pre-loading models improves performance for multiple comparisons
- Larger texts require more processing time
- GPU acceleration available if CUDA is installed

## Dependencies

- numpy: Numerical computations
- spacy: Text tokenization
- sentence-transformers: Semantic embeddings
- scikit-learn: Distance metrics and vectorization
- scipy: Statistical functions

## License

MIT License
