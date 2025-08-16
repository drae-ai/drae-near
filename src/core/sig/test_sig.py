import time
import unittest
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple
import json
from .sig import semantic_integrity_guarantee
import math


class SemanticIntegrityBenchmark(unittest.TestCase):
    """Comprehensive benchmarking suite for semantic integrity guarantee function."""

    # Class-level variable to store benchmark results across all test instances
    benchmark_results = []

    def setUp(self):
        """Initialize test resources."""
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.blank("en")

        # Test case pairs: (text1, text2, expected_behavior)
        self.test_cases = [
            # Identical texts
            ("The quick brown fox jumps over the lazy dog.",
             "The quick brown fox jumps over the lazy dog.",
             "identical"),

            # Similar texts with minor changes
            ("The quick brown fox jumps over the lazy dog.",
             "The quick brown fox leaps over the lazy dog.",
             "similar"),

            # Paraphrased texts
            ("The weather is beautiful today.",
             "Today's weather is lovely.",
             "paraphrase"),

            # Different topics
            ("The weather is beautiful today.",
             "Python is a programming language.",
             "different"),

            # Empty texts
            ("", "", "empty"),
            ("Hello world.", "", "one_empty"),

            # Very long texts
            ("This is a very long text that contains many words and sentences. " * 10,
             "This is another very long text with different content but similar length. " * 10,
             "long_texts"),

            # Technical content
            ("Machine learning algorithms require large datasets for training.",
             "Deep learning models need extensive data for effective training.",
             "technical"),

            # Different languages (mixed)
            ("Hello world", "Bonjour le monde", "different_language"),

            # Special characters
            ("Text with @#$% symbols!", "Text with symbols @#$%!", "special_chars"),
        ]

    def test_basic_functionality(self):
        """Test basic functionality with various input types."""
        print("\n=== Basic Functionality Tests ===")

        for i, (text1, text2, case_type) in enumerate(self.test_cases):
            with self.subTest(case=f"Case {i+1}: {case_type}"):
                start_time = time.time()
                result = semantic_integrity_guarantee(
                    text1, text2,
                    embedding_model=self.embedding_model,
                    spacy_nlp=self.nlp
                )
                execution_time = time.time() - start_time

                # Validate result structure
                self.assertIsInstance(result, dict)
                self.assertIn('cosine_distance', result)
                self.assertIn('js_divergence', result)
                self.assertIn('jaccard_similarity', result)

                # Validate value ranges
                if not np.isnan(result['cosine_distance']):
                    self.assertGreaterEqual(result['cosine_distance'], 0)
                    self.assertLessEqual(result['cosine_distance'], 1)

                if not np.isnan(result['js_divergence']):
                    self.assertGreaterEqual(result['js_divergence'], 0)
                    self.assertLessEqual(result['js_divergence'], 1)

                self.assertGreaterEqual(result['jaccard_similarity'], 0)
                self.assertLessEqual(result['jaccard_similarity'], 1)

                # Store benchmark data
                self.__class__.benchmark_results.append({
                    'case_type': case_type,
                    'text1_length': len(text1),
                    'text2_length': len(text2),
                    'execution_time': execution_time,
                    'cosine_distance': result['cosine_distance'],
                    'js_divergence': result['js_divergence'],
                    'jaccard_similarity': result['jaccard_similarity']
                })

                print(f"✓ {case_type}: {execution_time:.4f}s - "
                      f"cosine={result['cosine_distance']:.4f}, "
                      f"js={result['js_divergence']:.4f}, "
                      f"jaccard={result['jaccard_similarity']:.4f}")

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n=== Edge Cases Tests ===")

        edge_cases = [
            ("", "", "both_empty"),
            ("Hello", "", "second_empty"),
            ("", "World", "first_empty"),
            ("A" * 1000, "B" * 1000, "very_long_identical"),
            ("Single word", "Single word", "single_word"),
            ("Multiple\nlines\nhere", "Multiple\nlines\nhere", "multiline"),
        ]

        for text1, text2, case_type in edge_cases:
            with self.subTest(case=case_type):
                result = semantic_integrity_guarantee(
                    text1, text2,
                    embedding_model=self.embedding_model,
                    spacy_nlp=self.nlp
                )

                # Should not raise exceptions
                self.assertIsInstance(result, dict)
                print(f"✓ {case_type}: handled successfully")

    def test_parameter_variations(self):
        """Test different parameter combinations."""
        print("\n=== Parameter Variation Tests ===")

        base_text1 = "The quick brown fox jumps over the lazy dog."
        base_text2 = "A fast brown fox leaps over a lazy dog."

        # Test different n-gram ranges
        ngram_tests = [(1, 1), (1, 2), (2, 2), (1, 3)]
        for ngram_range in ngram_tests:
            with self.subTest(ngram_range=ngram_range):
                result = semantic_integrity_guarantee(
                    base_text1, base_text2,
                    embedding_model=self.embedding_model,
                    spacy_nlp=self.nlp,
                    ngram_range=ngram_range
                )
                self.assertIsInstance(result, dict)
                print(f"✓ ngram_range {ngram_range}: js_divergence={result['js_divergence']:.4f}")

        # Test different smoothing values
        smoothing_tests = [1e-8, 1e-6, 1e-4, 1e-2]
        for smoothing in smoothing_tests:
            with self.subTest(smoothing=smoothing):
                result = semantic_integrity_guarantee(
                    base_text1, base_text2,
                    embedding_model=self.embedding_model,
                    spacy_nlp=self.nlp,
                    smoothing=smoothing
                )
                self.assertIsInstance(result, dict)
                print(f"✓ smoothing {smoothing}: js_divergence={result['js_divergence']:.4f}")

    def test_performance_benchmark(self):
        """Benchmark performance with larger datasets."""
        print("\n=== Performance Benchmark ===")

        # Generate larger test texts
        base_text = "This is a sample text for performance testing. " * 50
        similar_text = "This is a sample text for performance evaluation. " * 50

        # Test multiple runs for consistency
        times = []
        for i in range(5):
            start_time = time.time()
            result = semantic_integrity_guarantee(
                base_text, similar_text,
                embedding_model=self.embedding_model,
                spacy_nlp=self.nlp
            )
            execution_time = time.time() - start_time
            times.append(execution_time)

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"Average execution time: {avg_time:.4f}s ± {std_time:.4f}s")
        print(f"Text length: {len(base_text)} characters")
        print(f"Results: cosine={result['cosine_distance']:.4f}, "
              f"js={result['js_divergence']:.4f}, "
              f"jaccard={result['jaccard_similarity']:.4f}")

        # Performance assertion (should complete within reasonable time)
        self.assertLess(avg_time, 5.0, "Performance test took too long")

    def test_consistency(self):
        """Test that results are consistent across multiple runs."""
        print("\n=== Consistency Tests ===")

        text1 = "The weather is beautiful today."
        text2 = "Today's weather is lovely."

        results = []
        for i in range(10):
            result = semantic_integrity_guarantee(
                text1, text2,
                embedding_model=self.embedding_model,
                spacy_nlp=self.nlp
            )
            results.append(result)

        # Check consistency of cosine distance
        cosine_distances = [r['cosine_distance'] for r in results]
        cosine_std = np.std(cosine_distances)

        # Check consistency of JS divergence
        js_divergences = [r['js_divergence'] for r in results]
        js_std = np.std(js_divergences)

        # Check consistency of Jaccard similarity
        jaccard_similarities = [r['jaccard_similarity'] for r in results]
        jaccard_std = np.std(jaccard_similarities)

        print(f"Cosine distance std: {cosine_std:.6f}")
        print(f"JS divergence std: {js_std:.6f}")
        print(f"Jaccard similarity std: {jaccard_std:.6f}")

        # Results should be very consistent (low standard deviation)
        self.assertLess(cosine_std, 1e-6, "Cosine distance not consistent")
        self.assertLess(js_std, 1e-6, "JS divergence not consistent")
        self.assertLess(jaccard_std, 1e-6, "Jaccard similarity not consistent")

    def test_metric_relationships(self):
        """Test expected relationships between metrics."""
        print("\n=== Metric Relationship Tests ===")

        # Test identical texts
        identical_result = semantic_integrity_guarantee(
            "Hello world", "Hello world",
            embedding_model=self.embedding_model,
            spacy_nlp=self.nlp
        )

        # Test very different texts
        different_result = semantic_integrity_guarantee(
            "Hello world", "Python programming language",
            embedding_model=self.embedding_model,
            spacy_nlp=self.nlp
        )

        # Identical texts should have low distances and high similarity
        self.assertLess(identical_result['cosine_distance'], 0.1)
        self.assertLess(identical_result['js_divergence'], 0.1)
        self.assertGreater(identical_result['jaccard_similarity'], 0.9)

        # Different texts should have higher distances and lower similarity
        self.assertGreater(different_result['cosine_distance'], identical_result['cosine_distance'])
        self.assertGreater(different_result['js_divergence'], identical_result['js_divergence'])
        self.assertLess(different_result['jaccard_similarity'], identical_result['jaccard_similarity'])

        print(f"Identical texts: cosine={identical_result['cosine_distance']:.4f}, "
              f"js={identical_result['js_divergence']:.4f}, "
              f"jaccard={identical_result['jaccard_similarity']:.4f}")
        print(f"Different texts: cosine={different_result['cosine_distance']:.4f}, "
              f"js={different_result['js_divergence']:.4f}, "
              f"jaccard={different_result['jaccard_similarity']:.4f}")

    def generate_benchmark_report(self):
        """Generate a comprehensive benchmark report."""
        if not self.__class__.benchmark_results:
            return

        print("\n" + "="*60)
        print("SEMANTIC INTEGRITY GUARANTEE BENCHMARK REPORT")
        print("="*60)

        # Group results by case type
        case_groups = {}
        for result in self.__class__.benchmark_results:
            case_type = result['case_type']
            if case_type not in case_groups:
                case_groups[case_type] = []
            case_groups[case_type].append(result)

        # Summary statistics
        print("\nSUMMARY STATISTICS:")
        print("-" * 40)

        for case_type, results in case_groups.items():
            avg_time = np.mean([r['execution_time'] for r in results])
            avg_cosine = np.mean([r['cosine_distance'] for r in results if not np.isnan(r['cosine_distance'])])
            avg_js = np.mean([r['js_divergence'] for r in results if not np.isnan(r['js_divergence'])])
            avg_jaccard = np.mean([r['jaccard_similarity'] for r in results])

            print(f"{case_type:20} | Time: {avg_time:.4f}s | "
                  f"Cosine: {avg_cosine:.4f} | JS: {avg_js:.4f} | "
                  f"Jaccard: {avg_jaccard:.4f}")

        # Overall performance
        all_times = [r['execution_time'] for r in self.__class__.benchmark_results]
        print(f"\nOVERALL PERFORMANCE:")
        print(f"Total test cases: {len(self.__class__.benchmark_results)}")
        print(f"Average execution time: {np.mean(all_times):.4f}s")
        print(f"Min execution time: {np.min(all_times):.4f}s")
        print(f"Max execution time: {np.max(all_times):.4f}s")
        print(f"Standard deviation: {np.std(all_times):.4f}s")

        # Save detailed results to JSON
        report_data = {
            'summary': {
                'total_cases': len(self.__class__.benchmark_results),
                'avg_execution_time': float(np.mean(all_times)),
                'min_execution_time': float(np.min(all_times)),
                'max_execution_time': float(np.max(all_times)),
                'std_execution_time': float(np.std(all_times))
            },
            'case_groups': case_groups,
            'detailed_results': self.__class__.benchmark_results
        }

        def clean_nans(obj):
            if isinstance(obj, float) and math.isnan(obj):
                return None
            elif isinstance(obj, dict):
                return {k: clean_nans(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nans(x) for x in obj]
            else:
                return obj

        report_data = clean_nans(report_data)
        with open('benchmark_report.json', 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\nDetailed results saved to: benchmark_report.json")


def run_benchmark():
    """Run the complete benchmark suite."""
    # Reset class-level benchmark results
    SemanticIntegrityBenchmark.benchmark_results = []

    # Create test suite
    suite = unittest.TestSuite()

    # Add all test methods
    test_methods = [
        'test_basic_functionality',
        'test_edge_cases',
        'test_parameter_variations',
        'test_performance_benchmark',
        'test_consistency',
        'test_metric_relationships'
    ]

    for method in test_methods:
        suite.addTest(SemanticIntegrityBenchmark(method))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Generate report
    benchmark = SemanticIntegrityBenchmark()
    benchmark.generate_benchmark_report()

    return result.wasSuccessful()


if __name__ == '__main__':
    print("Starting Semantic Integrity Guarantee Benchmark...")
    success = run_benchmark()

    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")

    exit(0 if success else 1)
