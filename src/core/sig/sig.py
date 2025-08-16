import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import jensenshannon
from typing import Optional, Dict

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

    Notes:
        - Cosine distance: 0 means identical, 1 means maximally different.
        - JS divergence: 0 means identical, 1 means maximally different (bounded).
        - Jaccard similarity: 1 means identical, 0 means no overlap.
        - If either text is empty, returns NaN for distances/divergence, 0 for similarity.
    """
    # --- Preprocessing ---
    nlp = spacy_nlp or spacy.blank("en")
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
    model = embedding_model or SentenceTransformer(embedding_model_name)
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
