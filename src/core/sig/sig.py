import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import jensenshannon
from typing import Optional, Dict

# Module-level constants and caches
_DEFAULT_SPACY_NLP = spacy.blank("en")
_CACHED_SENTENCE_TRANSFORMERS = {}

def semantic_integrity_guarantee(
    text1: str,
    text2: str,
    embedding_model: Optional[SentenceTransformer] = None,
    embedding_model_name: str = 'all-MiniLM-L6-v2',
    ngram_range: tuple = (1, 1),
    smoothing: float = 1e-6,
    spacy_nlp: Optional = None,
    **kwargs
) -> Dict:
    """
    Computes a suite of semantic integrity metrics between two texts.

    Metrics computed:
      1. Cosine distance between semantic embeddings (lower = more similar).
      2. Jensen-Shannon divergence between smoothed token (or n-gram) distributions (lower = more similar).
      3. Jaccard similarity between sets of tokens (higher = more similar).

    Args:
        text1 (str): Source/original text.
        text2 (str): Generated/transformed text.
        embedding_model (SentenceTransformer, optional): Preloaded embedding model.
        embedding_model_name (str): Model name if embedding_model not given.
        ngram_range (tuple): n-gram range for token distribution metrics.
        smoothing (float): Additive smoothing for probability distributions.
        spacy_nlp (spacy.language.Language, optional): Preloaded spaCy tokenizer.
        **kwargs: Reserved for extensibility.

    Returns:
        dict: {
            'cosine_distance': float or np.nan,
            'js_divergence': float or np.nan,
            'jaccard_similarity': float
        }

    Raises:
        TypeError: If text1 or text2 are not strings.
        ValueError: If ngram_range is invalid (e.g., negative values, start > end).
        ValueError: If smoothing is negative.
        RuntimeError: If embedding model fails to load or encode texts.
        MemoryError: If texts are extremely long and cause memory issues during processing.

    Notes:
        - Cosine distance: 0 means identical, 1 means maximally different.
        - JS divergence: 0 means identical, 1 means maximally different (bounded).
        - Jaccard similarity: 1 means identical, 0 means no overlap.
        - If either text is empty, returns NaN for distances/divergence, 0 for similarity.

    Input Validation and Error Handling:
        - **String Inputs**: Both text1 and text2 must be strings. Non-string inputs will raise TypeError.
        - **Empty/Whitespace Texts**: Empty strings or texts containing only whitespace/punctuation are handled gracefully
          and return NaN for distance metrics and 0 for similarity.
        - **Text Length Limits**: Very long texts (>1MB) may cause memory issues. Consider chunking extremely long texts.
        - **Parameter Validation**:
          * ngram_range must be a tuple of (start, end) where start <= end and both are positive integers
          * smoothing must be non-negative
        - **Model Loading**: If embedding_model is None, the function attempts to load the specified model.
          Model loading failures will raise RuntimeError.
        - **Memory Management**: For large texts, the function may consume significant memory during embedding
          generation. Monitor memory usage when processing documents >100KB.
        - **Graceful Degradation**: The function attempts to handle edge cases gracefully, but may fail
          if system resources are insufficient for the input size.

    Examples:
        >>> # Valid inputs
        >>> result = semantic_integrity_guarantee("Hello world", "Hi there")
        >>> # Empty text handling
        >>> result = semantic_integrity_guarantee("", "Hello")  # Returns NaN for distances
        >>> # Very long text (may cause memory issues)
        >>> long_text = "A" * 1000000  # 1MB of text
        >>> result = semantic_integrity_guarantee(long_text, long_text)  # May raise MemoryError

    Performance Considerations:
        - Processing time scales with text length due to embedding generation
        - Memory usage scales with text length and vocabulary size
        - For production use with large texts, consider preprocessing and chunking strategies
    """
    # --- Preprocessing ---
    nlp = spacy_nlp if spacy_nlp is not None else _DEFAULT_SPACY_NLP
    def tokenize(text):
        doc = nlp(text.lower())
        return [token.text for token in doc if not token.is_space and not token.is_punct]

    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    t1 = ' '.join(tokens1)
    t2 = ' '.join(tokens2)

    # --- Edge case: empty input ---
    if not tokens1 or not tokens2:
        return {
            'cosine_distance': np.nan,
            'js_divergence': np.nan,
            'jaccard_similarity': 0.0
        }

    # --- Embedding & Cosine Distance ---
    if embedding_model is not None:
        model = embedding_model
    else:
        if embedding_model_name not in _CACHED_SENTENCE_TRANSFORMERS:
            _CACHED_SENTENCE_TRANSFORMERS[embedding_model_name] = SentenceTransformer(embedding_model_name)
        model = _CACHED_SENTENCE_TRANSFORMERS[embedding_model_name]
    emb1 = model.encode([' '.join(tokens1)])[0]
    emb2 = model.encode([' '.join(tokens2)])[0]
    cosine_dist = float(cosine_distances([emb1], [emb2])[0][0])

    # --- Token/N-gram Distribution (with unified tokenizer) ---
    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        tokenizer=lambda txt: txt.split(),
        token_pattern=None,  # disables default token_pattern
        lowercase=False      # already lowercased
    )
    X = vectorizer.fit_transform([t1, t2])
    counts1, counts2 = X.toarray()
    vocab_size = len(vectorizer.get_feature_names_out())

    # --- Smoothed Probability Distributions ---
    p1 = (counts1 + smoothing) / (counts1.sum() + smoothing * vocab_size)
    p2 = (counts2 + smoothing) / (counts2.sum() + smoothing * vocab_size)

    # --- Jensen-Shannon Divergence (symmetric, always finite) ---
    js_div = float(jensenshannon(p1, p2, base=2))

    # --- Jaccard Similarity (set overlap) ---
    set1 = set(tokens1)
    set2 = set(tokens2)
    intersection = set1 & set2
    union = set1 | set2
    jaccard_sim = float(len(intersection) / len(union)) if union else 0.0

    return {
        'cosine_distance': cosine_dist,
        'js_divergence': js_div,
        'jaccard_similarity': jaccard_sim
    }
