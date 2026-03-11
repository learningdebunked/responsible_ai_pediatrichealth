"""Amazon Reviews 2023 Product Metadata Loader.

Loads product metadata from the Amazon Reviews 2023 dataset:
    https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

This module validates the DevelopMap taxonomy against real product data
by classifying Amazon Baby Products and Toys & Games into developmental
domains using the ProductClassifier, then reporting coverage and agreement
statistics.

Required input: Product metadata JSONL files for Baby_Products and
Toys_and_Games categories, downloaded from HuggingFace.

Usage:
    loader = AmazonProductLoader('data/raw/amazon/')
    report = loader.load_and_classify()
    loader.compute_taxonomy_agreement('data/raw/human_labels.csv')
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import warnings


class AmazonProductLoader:
    """Load Amazon product metadata and classify into developmental domains.

    Validates the DevelopMap taxonomy by running the ProductClassifier on
    real product titles/descriptions and reporting per-domain statistics.
    """

    def __init__(self, amazon_data_dir: str, use_embeddings: bool = True):
        """Initialize Amazon product loader.

        Args:
            amazon_data_dir: Path to directory containing Amazon metadata files.
                Expected files: meta_Baby_Products.jsonl, meta_Toys_and_Games.jsonl
            use_embeddings: Whether to use semantic embeddings in classification
        """
        self.data_dir = Path(amazon_data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Amazon data directory not found: {self.data_dir}\n"
                f"Download from: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
            )
        self.use_embeddings = use_embeddings
        self.products_df: Optional[pd.DataFrame] = None
        self.classified_df: Optional[pd.DataFrame] = None
        self._classifier = None

    def _get_classifier(self):
        """Lazy-load the ProductClassifier to avoid import issues."""
        if self._classifier is None:
            from ..taxonomy.classifier import ProductClassifier
            self._classifier = ProductClassifier(use_embeddings=self.use_embeddings)
        return self._classifier

    def load_and_classify(self) -> Dict:
        """Load Amazon products and classify each into developmental domains.

        Returns:
            Dict with classification report:
                - per_domain_counts: products per domain
                - confidence_distributions: score distributions per domain
                - unmapped_rate: fraction of products with no domain match
                - ambiguous_rate: fraction mapped to 3+ domains
                - total_products: total products loaded
        """
        self._load_products()
        self._classify_products()
        report = self._generate_report()
        return report

    def _load_products(self):
        """Load product metadata from JSONL files."""
        jsonl_patterns = [
            'meta_Baby_Products.jsonl',
            'meta_Toys_and_Games.jsonl',
            'meta_Baby.jsonl',
            'meta_Toys.jsonl',
        ]

        dfs = []
        for pattern in jsonl_patterns:
            filepath = self.data_dir / pattern
            if filepath.exists():
                print(f"Loading {filepath.name}...")
                try:
                    df = pd.read_json(filepath, lines=True)
                    dfs.append(df)
                    print(f"  Loaded {len(df)} products")
                except Exception as e:
                    warnings.warn(f"Could not load {filepath}: {e}")

        # Also try gzipped versions
        for pattern in jsonl_patterns:
            gz_path = self.data_dir / (pattern + '.gz')
            if gz_path.exists():
                print(f"Loading {gz_path.name}...")
                try:
                    df = pd.read_json(gz_path, lines=True, compression='gzip')
                    dfs.append(df)
                    print(f"  Loaded {len(df)} products")
                except Exception as e:
                    warnings.warn(f"Could not load {gz_path}: {e}")

        if not dfs:
            raise FileNotFoundError(
                f"No Amazon product metadata files found in {self.data_dir}. "
                f"Expected files like meta_Baby_Products.jsonl or meta_Toys_and_Games.jsonl. "
                f"Download from: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
            )

        self.products_df = pd.concat(dfs, ignore_index=True)

        # Extract relevant columns
        text_col = None
        for candidate in ['title', 'name', 'product_title']:
            if candidate in self.products_df.columns:
                text_col = candidate
                break

        if text_col is None:
            raise ValueError("No title/name column found in Amazon data")

        desc_col = None
        for candidate in ['description', 'details', 'feature']:
            if candidate in self.products_df.columns:
                desc_col = candidate
                break

        # Combine title + description for classification
        self.products_df['product_text'] = self.products_df[text_col].fillna('')
        if desc_col:
            desc_text = self.products_df[desc_col].apply(
                lambda x: ' '.join(x) if isinstance(x, list) else str(x) if pd.notna(x) else ''
            )
            self.products_df['product_text'] = (
                self.products_df['product_text'] + ' ' + desc_text
            )

        # Keep Amazon category for validation
        cat_col = None
        for candidate in ['main_category', 'category', 'categories']:
            if candidate in self.products_df.columns:
                cat_col = candidate
                break

        if cat_col:
            self.products_df['amazon_category'] = self.products_df[cat_col].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else str(x) if pd.notna(x) else ''
            )
        else:
            self.products_df['amazon_category'] = ''

        # Drop products with empty text
        self.products_df = self.products_df[
            self.products_df['product_text'].str.strip().str.len() > 0
        ].reset_index(drop=True)

        print(f"\nTotal products loaded: {len(self.products_df)}")

    def _classify_products(self):
        """Run ProductClassifier on all loaded products."""
        classifier = self._get_classifier()
        n = len(self.products_df)
        print(f"\nClassifying {n} products into developmental domains...")

        all_scores = []
        primary_domains = []
        n_domains_matched = []

        batch_size = 100
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_texts = self.products_df['product_text'].iloc[start:end].tolist()

            for text in batch_texts:
                scores = classifier.classify(text, threshold=0.3)
                all_scores.append(scores)

                if scores:
                    primary = max(scores, key=scores.get)
                    primary_domains.append(primary)
                    n_domains_matched.append(len(scores))
                else:
                    primary_domains.append(None)
                    n_domains_matched.append(0)

            if end % 1000 == 0 or end == n:
                print(f"  Classified {end}/{n} products")

        self.products_df['primary_domain'] = primary_domains
        self.products_df['n_domains'] = n_domains_matched
        self.products_df['domain_scores'] = all_scores
        self.classified_df = self.products_df

    def _generate_report(self) -> Dict:
        """Generate classification report."""
        df = self.classified_df
        n = len(df)

        # Per-domain counts
        domain_counts = df['primary_domain'].value_counts().to_dict()

        # Confidence distributions per domain
        confidence_dists = {}
        for domain in domain_counts:
            if domain is None:
                continue
            scores = [
                s.get(domain, 0.0)
                for s in df['domain_scores']
                if domain in s
            ]
            if scores:
                confidence_dists[domain] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'n': len(scores),
                }

        # Unmapped and ambiguous rates
        unmapped = (df['primary_domain'].isna()).sum()
        ambiguous = (df['n_domains'] >= 3).sum()

        report = {
            'per_domain_counts': domain_counts,
            'confidence_distributions': confidence_dists,
            'unmapped_rate': float(unmapped / n) if n > 0 else 0.0,
            'ambiguous_rate': float(ambiguous / n) if n > 0 else 0.0,
            'total_products': n,
            'unmapped_count': int(unmapped),
            'ambiguous_count': int(ambiguous),
        }

        self._print_report(report)
        return report

    def _print_report(self, report: Dict):
        """Print classification report."""
        print(f"\n{'='*60}")
        print("AMAZON PRODUCT CLASSIFICATION REPORT")
        print(f"{'='*60}")
        print(f"Total products: {report['total_products']}")
        print(f"Unmapped: {report['unmapped_count']} ({report['unmapped_rate']*100:.1f}%)")
        print(f"Ambiguous (3+ domains): {report['ambiguous_count']} ({report['ambiguous_rate']*100:.1f}%)")

        print(f"\nProducts per domain:")
        for domain, count in sorted(report['per_domain_counts'].items(),
                                     key=lambda x: x[1], reverse=True):
            if domain is not None:
                conf = report['confidence_distributions'].get(domain, {})
                mean_conf = conf.get('mean', 0)
                print(f"  {str(domain):20s}: {count:6d} (mean confidence: {mean_conf:.3f})")

        print(f"{'='*60}\n")

    def compute_taxonomy_agreement(self, human_labels_path: str) -> Dict:
        """Compute Cohen's kappa agreement vs human labels.

        Args:
            human_labels_path: Path to CSV with columns:
                product_text, human_domain (ground truth domain label)

        Returns:
            Dict with kappa, per-domain precision/recall, confusion matrix
        """
        from sklearn.metrics import cohen_kappa_score, classification_report

        labels_df = pd.read_csv(human_labels_path)

        if 'product_text' not in labels_df.columns or 'human_domain' not in labels_df.columns:
            raise ValueError(
                "Human labels CSV must have columns: product_text, human_domain"
            )

        classifier = self._get_classifier()

        predicted = []
        for text in labels_df['product_text']:
            result = classifier.get_primary_domain(text)
            predicted.append(result[0] if result else 'unmapped')

        labels_df['predicted_domain'] = predicted
        human = labels_df['human_domain'].values
        pred = labels_df['predicted_domain'].values

        kappa = cohen_kappa_score(human, pred)
        report = classification_report(human, pred, output_dict=True, zero_division=0)

        result = {
            'cohens_kappa': float(kappa),
            'classification_report': report,
            'n_samples': len(labels_df),
        }

        print(f"\nCohen's Kappa: {kappa:.3f}")
        print(classification_report(human, pred, zero_division=0))

        return result

    def get_product_embeddings_by_domain(self) -> Dict[str, np.ndarray]:
        """Compute average product embeddings per domain for visualization.

        Returns:
            Dict mapping domain name to mean embedding vector
        """
        if self.classified_df is None:
            raise RuntimeError("Call load_and_classify() first")

        classifier = self._get_classifier()
        if classifier.model is None:
            raise RuntimeError("Classifier must use embeddings for this method")

        domain_embeddings = {}
        df = self.classified_df

        for domain in df['primary_domain'].dropna().unique():
            domain_texts = df[df['primary_domain'] == domain]['product_text'].tolist()

            # Sample if too many
            if len(domain_texts) > 500:
                rng = np.random.RandomState(42)
                domain_texts = list(rng.choice(domain_texts, 500, replace=False))

            if domain_texts:
                embeddings = classifier.model.encode(domain_texts, convert_to_numpy=True,
                                                      show_progress_bar=False)
                domain_embeddings[domain] = embeddings.mean(axis=0)

        return domain_embeddings

    def save_report(self, output_path: str, report: Dict):
        """Save classification report to JSON.

        Args:
            output_path: Path to save report
            report: Report dict from load_and_classify()
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Make serializable
        serializable = {
            k: v for k, v in report.items()
            if k != 'per_domain_counts' or not any(v2 is None for v2 in v)
        }
        # Convert None keys
        if 'per_domain_counts' in report:
            serializable['per_domain_counts'] = {
                str(k): v for k, v in report['per_domain_counts'].items()
            }

        with open(output, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved report to {output}")
