#!/usr/bin/env python3
"""
Comprehensive benchmark for sig_guard using multiple datasets:
1. STS Benchmark (STSb) - for ROC-style analysis
2. PAWS-Wiki/PAWS-QQP - for adversarial pair detection
3. PARAPHRASUS - for fine-grained paraphrase detection
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from .sig import semantic_integrity_guarantee
try:
    from .sig import semantic_integrity_guarantee
    from .sig_guard import passes_sig_guard, DELTA
except (ImportError, ModuleNotFoundError):
    from sig import semantic_integrity_guarantee
    from sig_guard import passes_sig_guard, DELTA

class SIGGuardBenchmark:
    """Comprehensive benchmark for semantic integrity guard."""

    def __init__(self):
        """Initialize the benchmark with models and datasets."""
        print("🚀 Initializing SIG Guard Benchmark...")

        # Initialize models
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.blank("en")

        # Store results
        self.results = {
            'sts_benchmark': {},
            'paws': {},
            'paraphrasus': {},
            'summary': {}
        }

        print("✅ Models loaded successfully")

    def load_sts_benchmark(self) -> List[Tuple[str, str, float]]:
        """Load STS Benchmark dataset."""
        print("📊 Loading STS Benchmark dataset...")

        try:
            # Load STS Benchmark
            dataset = load_dataset("glue", "stsb", split="validation")

            # Convert to our format: (text1, text2, similarity_score)
            pairs = []
            for item in dataset:
                text1 = item['sentence1']
                text2 = item['sentence2']
                score = item['label'] / 5.0  # Normalize to 0-1
                pairs.append((text1, text2, score))

            print(f"✅ Loaded {len(pairs)} STS Benchmark pairs")
            return pairs

        except Exception as e:
            print(f"❌ Error loading STS Benchmark: {e}")
            # Fallback to a small sample for testing
            return [
                ("A man is playing guitar.", "A man is playing a guitar.", 0.9),
                ("A man is playing guitar.", "A woman is singing.", 0.3),
                ("The weather is nice today.", "Today's weather is pleasant.", 0.8),
                ("The weather is nice today.", "I love programming.", 0.1),
            ]

    def load_paws_dataset(self) -> List[Tuple[str, str, bool]]:
        """Load PAWS dataset for paraphrase detection."""
        print("📊 Loading PAWS dataset...")

        try:
            # Load PAWS-Wiki
            dataset = load_dataset("paws", "labeled_final", split="test")

            # Convert to our format: (text1, text2, is_paraphrase)
            pairs = []
            for item in dataset:
                text1 = item['sentence1']
                text2 = item['sentence2']
                is_paraphrase = bool(item['label'])
                pairs.append((text1, text2, is_paraphrase))

            # Limit to first 1000 for performance
            pairs = pairs[:1000]
            print(f"✅ Loaded {len(pairs)} PAWS pairs")
            return pairs

        except Exception as e:
            print(f"❌ Error loading PAWS: {e}")
            # Fallback examples
            return [
                ("The cat sat on the mat.", "The cat is sitting on the mat.", True),
                ("The cat sat on the mat.", "The dog ran in the park.", False),
                ("I love this movie.", "I really enjoy this film.", True),
                ("I love this movie.", "The weather is terrible.", False),
            ]

    def load_paraphrasus_dataset(self) -> List[Tuple[str, str, str]]:
        """Load PARAPHRASUS dataset for fine-grained analysis."""
        print("📊 Loading PARAPHRASUS dataset...")

        try:
            # Try to load PARAPHRASUS (this might not be available yet)
            # For now, we'll create synthetic data that mimics the structure
            pairs = []

            # Generate synthetic PARAPHRASUS-like data
            base_sentences = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "The weather forecast predicts rain tomorrow.",
                "Python is a popular programming language.",
                "The restaurant serves delicious Italian food."
            ]

            # Create different types of paraphrases
            for base in base_sentences:
                # Exact paraphrase
                pairs.append((base, base, "exact"))

                # Synonym paraphrase
                if "quick" in base:
                    pairs.append((base, base.replace("quick", "fast"), "synonym"))
                elif "delicious" in base:
                    pairs.append((base, base.replace("delicious", "tasty"), "synonym"))

                # Structural paraphrase
                if "Machine learning is" in base:
                    pairs.append(("Machine learning is a subset of artificial intelligence.",
                                 "Artificial intelligence includes machine learning as a component.", "structural"))

                # Different meaning
                pairs.append((base, "The sky is blue and the grass is green.", "different"))

            print(f"✅ Generated {len(pairs)} PARAPHRASUS-like pairs")
            return pairs

        except Exception as e:
            print(f"❌ Error loading PARAPHRASUS: {e}")
            return []

    def compute_metrics(self, text1: str, text2: str) -> Dict[str, float]:
        """Compute semantic integrity metrics for a text pair."""
        try:
            return semantic_integrity_guarantee(
                text1, text2,
                embedding_model=self.embedding_model,
                spacy_nlp=self.nlp
            )
        except Exception as e:
            print(f"❌ Error computing metrics: {e}")
            return {
                'cosine_distance': np.nan,
                'js_divergence': np.nan,
                'jaccard_similarity': np.nan
            }

    def benchmark_sts(self, pairs: List[Tuple[str, str, float]]):
        """Benchmark using STS Benchmark for ROC analysis."""
        print("\n🔍 Running STS Benchmark Analysis...")

        results = []
        baseline_metrics = None

        for i, (text1, text2, gold_score) in enumerate(pairs):
            if i % 100 == 0:
                print(f"  Processing pair {i}/{len(pairs)}")

            # Compute metrics
            metrics = self.compute_metrics(text1, text2)

            # Use first pair as baseline
            if baseline_metrics is None:
                baseline_metrics = metrics.copy()

            # Check if guard passes
            passes = passes_sig_guard(baseline_metrics, metrics)

            results.append({
                'text1': text1,
                'text2': text2,
                'gold_score': gold_score,
                'cosine_distance': metrics['cosine_distance'],
                'js_divergence': metrics['js_divergence'],
                'jaccard_similarity': metrics['jaccard_similarity'],
                'guard_passes': passes
            })

        # Analyze results
        df = pd.DataFrame(results)

        # Create bands for analysis
        df['similarity_band'] = pd.cut(df['gold_score'],
                                     bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                     labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

        # Calculate pass rates by band
        pass_rates = df.groupby('similarity_band')['guard_passes'].agg(['mean', 'count'])

        # ROC analysis using gold score as ground truth
        # Higher gold score = should pass more often
        y_true = (df['gold_score'] > 0.5).astype(int)
        y_score = df['jaccard_similarity']  # Use Jaccard as proxy for similarity

        try:
            roc_auc = roc_auc_score(y_true, y_score)
        except:
            roc_auc = np.nan

        self.results['sts_benchmark'] = {
            'total_pairs': len(pairs),
            'pass_rates_by_band': pass_rates.to_dict(),
            'roc_auc': roc_auc,
            'mean_metrics': {
                'cosine_distance': df['cosine_distance'].mean(),
                'js_divergence': df['js_divergence'].mean(),
                'jaccard_similarity': df['jaccard_similarity'].mean()
            },
            'sample_results': results[:10]  # Store first 10 for inspection
        }

        print(f"✅ STS Benchmark completed: {len(pairs)} pairs analyzed")
        print(f"   ROC AUC: {roc_auc:.3f}")
        print(f"   Pass rates by similarity band:")
        for band, stats in pass_rates.iterrows():
            print(f"     {band}: {stats['mean']:.2%} ({stats['count']} pairs)")

    def benchmark_paws(self, pairs: List[Tuple[str, str, bool]]):
        """Benchmark using PAWS for adversarial detection."""
        print("\n🔍 Running PAWS Adversarial Detection Analysis...")

        results = []
        baseline_metrics = None

        for i, (text1, text2, is_paraphrase) in enumerate(pairs):
            if i % 100 == 0:
                print(f"  Processing pair {i}/{len(pairs)}")

            # Compute metrics
            metrics = self.compute_metrics(text1, text2)

            # Use first pair as baseline
            if baseline_metrics is None:
                baseline_metrics = metrics.copy()

            # Check if guard passes
            passes = passes_sig_guard(baseline_metrics, metrics)

            results.append({
                'text1': text1,
                'text2': text2,
                'is_paraphrase': is_paraphrase,
                'cosine_distance': metrics['cosine_distance'],
                'js_divergence': metrics['js_divergence'],
                'jaccard_similarity': metrics['jaccard_similarity'],
                'guard_passes': passes
            })

        # Analyze results
        df = pd.DataFrame(results)

        # Calculate performance metrics
        y_true = df['is_paraphrase'].astype(int)
        y_pred = df['guard_passes'].astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Analyze by metric thresholds
        cosine_threshold = 0.3
        js_threshold = 0.5
        jaccard_threshold = 0.7

        df['cosine_fails'] = df['cosine_distance'] > cosine_threshold
        df['js_fails'] = df['js_divergence'] > js_threshold
        df['jaccard_fails'] = df['jaccard_similarity'] < jaccard_threshold

        self.results['paws'] = {
            'total_pairs': len(pairs),
            'paraphrase_pairs': df['is_paraphrase'].sum(),
            'non_paraphrase_pairs': (~df['is_paraphrase']).sum(),
            'performance_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'confusion_matrix': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            },
            'metric_analysis': {
                'cosine_fail_rate': df['cosine_fails'].mean(),
                'js_fail_rate': df['js_fails'].mean(),
                'jaccard_fail_rate': df['jaccard_fails'].mean()
            },
            'sample_results': results[:10]
        }

        print(f"✅ PAWS Benchmark completed: {len(pairs)} pairs analyzed")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1 Score: {f1:.3f}")
        print(f"   Paraphrase pairs: {df['is_paraphrase'].sum()}")
        print(f"   Non-paraphrase pairs: {(~df['is_paraphrase']).sum()}")

    def benchmark_paraphrasus(self, pairs: List[Tuple[str, str, str]]):
        """Benchmark using PARAPHRASUS for fine-grained analysis."""
        print("\n🔍 Running PARAPHRASUS Fine-grained Analysis...")

        results = []
        baseline_metrics = None

        for i, (text1, text2, paraphrase_type) in enumerate(pairs):
            if i % 50 == 0:
                print(f"  Processing pair {i}/{len(pairs)}")

            # Compute metrics
            metrics = self.compute_metrics(text1, text2)

            # Use first pair as baseline
            if baseline_metrics is None:
                baseline_metrics = metrics.copy()

            # Check if guard passes
            passes = passes_sig_guard(baseline_metrics, metrics)

            results.append({
                'text1': text1,
                'text2': text2,
                'paraphrase_type': paraphrase_type,
                'cosine_distance': metrics['cosine_distance'],
                'js_divergence': metrics['js_divergence'],
                'jaccard_similarity': metrics['jaccard_similarity'],
                'guard_passes': passes
            })

        # Analyze results
        df = pd.DataFrame(results)

        # Analyze by paraphrase type
        type_analysis = df.groupby('paraphrase_type').agg({
            'guard_passes': ['mean', 'count'],
            'cosine_distance': 'mean',
            'js_divergence': 'mean',
            'jaccard_similarity': 'mean'
        }).round(4)

        # Calculate consistency score
        # For exact/synonym: should pass, for different: should fail
        expected_passes = df['paraphrase_type'].isin(['exact', 'synonym', 'structural'])
        consistency = (df['guard_passes'] == expected_passes).mean()

        self.results['paraphrasus'] = {
            'total_pairs': len(pairs),
            'consistency_score': consistency,
            'type_analysis': type_analysis.to_dict(),
            'paraphrase_type_counts': df['paraphrase_type'].value_counts().to_dict(),
            'sample_results': results[:10]
        }

        print(f"✅ PARAPHRASUS Benchmark completed: {len(pairs)} pairs analyzed")
        print(f"   Consistency Score: {consistency:.3f}")
        print(f"   Analysis by paraphrase type:")
        for paraphrase_type in df['paraphrase_type'].unique():
            subset = df[df['paraphrase_type'] == paraphrase_type]
            pass_rate = subset['guard_passes'].mean()
            count = len(subset)
            print(f"     {paraphrase_type}: {pass_rate:.2%} ({count} pairs)")

    def generate_summary(self):
        """Generate overall summary of benchmark results."""
        print("\n📋 Generating Summary Report...")

        summary = {
            'benchmark_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'delta_settings': DELTA,
                'embedding_model': 'all-MiniLM-L6-v2'
            },
            'dataset_summary': {
                'sts_benchmark': {
                    'total_pairs': self.results['sts_benchmark'].get('total_pairs', 0),
                    'roc_auc': self.results['sts_benchmark'].get('roc_auc', np.nan)
                },
                'paws': {
                    'total_pairs': self.results['paws'].get('total_pairs', 0),
                    'accuracy': self.results['paws'].get('performance_metrics', {}).get('accuracy', np.nan)
                },
                'paraphrasus': {
                    'total_pairs': self.results['paraphrasus'].get('total_pairs', 0),
                    'consistency': self.results['paraphrasus'].get('consistency_score', np.nan)
                }
            },
            'recommendations': []
        }

        # Generate recommendations based on results
        if 'sts_benchmark' in self.results and self.results['sts_benchmark']:
            roc_auc = self.results['sts_benchmark'].get('roc_auc', 0)
            if roc_auc > 0.8:
                summary['recommendations'].append("✅ STS Benchmark: Guard shows good discriminative power")
            elif roc_auc > 0.6:
                summary['recommendations'].append("⚠️ STS Benchmark: Guard shows moderate discriminative power")
            else:
                summary['recommendations'].append("❌ STS Benchmark: Guard shows poor discriminative power")

        if 'paws' in self.results and self.results['paws']:
            accuracy = self.results['paws'].get('performance_metrics', {}).get('accuracy', 0)
            if accuracy > 0.8:
                summary['recommendations'].append("✅ PAWS: Guard effectively detects adversarial pairs")
            elif accuracy > 0.6:
                summary['recommendations'].append("⚠️ PAWS: Guard moderately detects adversarial pairs")
            else:
                summary['recommendations'].append("❌ PAWS: Guard struggles with adversarial pairs")

        if 'paraphrasus' in self.results and self.results['paraphrasus']:
            consistency = self.results['paraphrasus'].get('consistency_score', 0)
            if consistency > 0.8:
                summary['recommendations'].append("✅ PARAPHRASUS: Guard shows consistent behavior across paraphrase types")
            elif consistency > 0.6:
                summary['recommendations'].append("⚠️ PARAPHRASUS: Guard shows moderate consistency")
            else:
                summary['recommendations'].append("❌ PARAPHRASUS: Guard shows inconsistent behavior")

        self.results['summary'] = summary

        print("✅ Summary generated")
        print("\n📊 SUMMARY:")
        for rec in summary['recommendations']:
            print(f"   {rec}")

    def save_results(self, filename: str = 'sig_guard_benchmark_results.json'):
        """Save benchmark results to file."""
        print(f"\n💾 Saving results to {filename}...")

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(key): convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, tuple):
                return str(obj)  # Convert tuples to strings
            else:
                return obj

        results_json = convert_numpy(self.results)

        with open(filename, 'w') as f:
            json.dump(results_json, f, indent=2)

        print(f"✅ Results saved to {filename}")

    def run_full_benchmark(self):
        """Run the complete benchmark suite."""
        print("🚀 Starting SIG Guard Comprehensive Benchmark")
        print("=" * 60)

        start_time = time.time()

        try:
            # Load datasets
            sts_pairs = self.load_sts_benchmark()
            paws_pairs = self.load_paws_dataset()
            paraphrasus_pairs = self.load_paraphrasus_dataset()

            # Run benchmarks
            if sts_pairs:
                self.benchmark_sts(sts_pairs)

            if paws_pairs:
                self.benchmark_paws(paws_pairs)

            if paraphrasus_pairs:
                self.benchmark_paraphrasus(paraphrasus_pairs)

            # Generate summary
            self.generate_summary()

            # Save results
            self.save_results()

            total_time = time.time() - start_time
            print(f"\n🎉 Benchmark completed in {total_time:.2f} seconds!")

            return True

        except Exception as e:
            print(f"\n❌ Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main entry point for the benchmark."""
    benchmark = SIGGuardBenchmark()
    success = benchmark.run_full_benchmark()

    if success:
        print("\n📊 Check 'sig_guard_benchmark_results.json' for detailed results")
        return 0
    else:
        print("\n❌ Benchmark failed")
        return 1


if __name__ == '__main__':
    exit(main())
