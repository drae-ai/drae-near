## Justification and Illustration of Metrics in `semantic_integrity_guarantee`

The `semantic_integrity_guarantee` function is designed to robustly assess how much meaning and information is preserved between two texts—such as
an original and a transformed or generated version. To do this, it combines three complementary metrics, each capturing a different aspect of
semantic similarity or divergence:

---

### 1. **Cosine Distance (on Semantic Embeddings)**

**What it measures:**
Cosine distance quantifies the semantic similarity between two texts by comparing their dense vector embeddings (from a neural language model). A
value of 0 means the texts are semantically identical; 1 means they are maximally different.

**Why it is valuable:**
- Captures deep, contextual meaning beyond surface word overlap.
- Robust to paraphrasing or rewording that preserves meaning.

**Caveats:**
- The reliability of cosine distance depends heavily on the quality and domain-relevance of the underlying embedding model. Poorly chosen or
outdated models may yield misleading results.
- Subtle semantic shifts (e.g., negation, factual errors) may not always be detected.

**Example:**

| Text 1                              | Text 2                              | Cosine Distance | Interpretation              |
|-------------------------------------|-------------------------------------|-----------------|-----------------------------|
| "The cat sat on the mat."           | "A feline was sitting on the rug."  | Low (~0.1)      | Semantically very similar   |
| "The cat sat on the mat."           | "Quantum mechanics is complicated." | High (~0.9)     | Semantically unrelated      |
| "The cat sat on the mat."           | "The cat did not sit on the mat."   | Low (~0.2)      | High similarity despite opposite factual meaning |

---

### 2. **Jensen-Shannon Divergence (on Token/N-gram Distributions)**

**What it measures:**
Jensen-Shannon (JS) divergence measures the similarity between the probability distributions of tokens (or n-grams) in the two texts, after
smoothing. A value of 0 means the distributions are identical; values approach 1 as distributions diverge.

**Why it is valuable:**
- Sensitive to changes in word or phrase usage.
- Detects if the transformed text omits or overemphasizes certain terms.
- Useful for catching subtle distributional shifts (e.g., topic drift, hallucination).

**Caveats:**
- For very short texts, JS divergence and Jaccard similarity may be highly correlated, as both reflect token-level overlap.
- Tokenization choices (especially for morphologically rich or non-segmented languages) can significantly affect results.

**Example:**

| Text 1                              | Text 2                              | JS Divergence   | Interpretation              |
|-------------------------------------|-------------------------------------|-----------------|-----------------------------|
| "The cat sat on the mat."           | "The cat sat on the mat."           | 0.0             | Identical token usage       |
| "The cat sat on the mat."           | "The dog barked at the mailman."    | High (~0.8-1.0) | Very different distributions|

---

### 3. **Jaccard Similarity (on Token Sets)**

**What it measures:**
Jaccard similarity quantifies the overlap between the sets of unique tokens in the two texts. A value of 1 means perfect overlap; 0 means no shared
tokens.

**Why it is valuable:**
- Simple, interpretable measure of surface-level lexical overlap.
- Useful for detecting deletion, addition, or replacement of key words.

**Caveats:**
- May be redundant with JS divergence for short texts.
- Sensitive to tokenization and normalization (e.g., punctuation, case, stemming, language-specific scripts).
- Does not account for word order or context.

**Example:**

| Text 1                              | Text 2                              | Jaccard Similarity | Interpretation              |
|-------------------------------------|-------------------------------------|--------------------|-----------------------------|
| "The cat sat on the mat."           | "The cat sat on the mat."           | 1.0                | All words overlap           |
| "The cat sat on the mat."           | "The dog barked at the mailman."    | 0.0                | No words overlap            |
| "The cat sat on the mat."           | "The cat lay on the mat."           | 0.83               | Most words overlap          |

---

## Rationale for Multi-Metric Approach

- **Comprehensiveness:** Each metric captures a different facet of semantic preservation—deep meaning (cosine), distributional similarity (JS), and
surface overlap (Jaccard).
- **Robustness:** Using multiple, complementary metrics reduces the risk of false positives/negatives that might arise from relying on a single
metric. However, for short texts or highly similar distributions, JS divergence and Jaccard similarity may be somewhat redundant; consider this when
interpreting results.
- **Transparency and Interpretability:** Users can see *why* two texts are judged similar or different, and can target improvements accordingly.
- **Extensibility:** For some tasks (e.g., translation or summarization), additional metrics like BLEU, ROUGE, or METEOR may provide further
insight.

**Example of Combined Interpretation:**

| Metric              | Value      | Interpretation                                     |
|---------------------|------------|----------------------------------------------------|
| Cosine Distance     | 0.12       | Highly similar meaning                             |
| JS Divergence       | 0.05       | Nearly identical token usage                       |
| Jaccard Similarity  | 0.90       | Most words overlap                                 |

*Conclusion:* The texts are semantically and lexically very similar—likely a faithful paraphrase.

---

## Edge Cases and Guarantees

- If either input is empty or contains only whitespace/punctuation, the function returns `NaN` for distances/divergence and `0` for similarity,
signaling that no meaningful comparison can be made.
- **Input Normalization:** It is recommended to normalize inputs (e.g., lowercasing, removing extraneous whitespace, standardizing punctuation) and
use appropriate tokenization, especially for languages with complex scripts or morphology.

---

## Limitations and Best Practices

- **Metric Sensitivity:** High cosine similarity does not guarantee perfect factual consistency or logical equivalence. For example, negation or
antonyms may not be detected.
- **Embedding Model Dependency:** The quality and domain of the embedding model used for cosine distance are crucial; use up-to-date,
domain-appropriate models.
- **Tokenization and Language:** Tokenization has a large effect, especially for languages like Chinese or Arabic. Use language-specific
preprocessing where needed.
- **Thresholds and Interpretation:** Typical thresholds (e.g., cosine distance < 0.2, Jaccard similarity > 0.8) can indicate strong similarity, but
should be calibrated for your dataset and use case.
- **Redundancy:** For very short texts, JS divergence and Jaccard similarity may offer overlapping information; in such cases, consider whether both
are necessary.
- **Critical Applications:** For high-stakes or factual-critical scenarios, supplement these metrics with human review or specialized factual
consistency checks.

---

## Additional Considerations

- **Metric Selection:** Depending on your application, you may wish to add or substitute metrics (e.g., BLEU, ROUGE, METEOR) for more nuanced
evaluation.
- **Interpretation in Context:** Always interpret metric values in context. For example, high Jaccard similarity with high JS divergence may signal
synonym replacement or topic drift.
- **Human-in-the-Loop:** Automated metrics are best used as guides; combine them with human judgment for important decisions.

---

## Summary

By combining **cosine distance**, **JS divergence**, and **Jaccard similarity**, `semantic_integrity_guarantee` provides a multidimensional,
transparent, and actionable evaluation of semantic integrity between texts. This approach maximizes reliability and interpretability, supporting
robust decision-making in text generation, summarization, translation, and related applications. However, be mindful of each metric’s limitations
and ensure preprocessing, model selection, and interpretation are tailored to your specific context and language.

---

**References:**
- [Cosine Similarity/Distance](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Jensen-Shannon Divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)
- [Jaccard Similarity](https://en.wikipedia.org/wiki/Jaccard_index)
