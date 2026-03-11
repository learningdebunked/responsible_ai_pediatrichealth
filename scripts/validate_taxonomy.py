#!/usr/bin/env python
"""Validate DevelopMap taxonomy against real Amazon product data.

Loads Amazon product metadata, runs ProductClassifier on each product,
computes precision/recall per domain, and generates validation figures.

Usage:
    python scripts/validate_taxonomy.py \
        --amazon_dir data/raw/amazon \
        --output_dir results/taxonomy \
        --fig_dir figures/taxonomy
"""

import argparse
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.taxonomy.developmap import DEVELOPMAP
from src.taxonomy.classifier import ProductClassifier


def validate_asq3_alignment(output_dir: Path):
    """Run ASQ-3 domain alignment validation."""
    print("\n" + "=" * 60)
    print("ASQ-3 ALIGNMENT VALIDATION")
    print("=" * 60)

    results = DEVELOPMAP.validate_against_asq3()

    with open(output_dir / 'asq3_alignment.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def validate_clinical_citations(output_dir: Path):
    """Verify all domains have specific clinical citations."""
    print("\n" + "=" * 60)
    print("CLINICAL CITATION VALIDATION")
    print("=" * 60)

    alignments = DEVELOPMAP.get_clinical_alignments()
    results = {}

    for domain, citation in alignments.items():
        has_year = any(c.isdigit() for c in citation)
        has_author = '(' in citation and ')' in citation
        has_instrument = len(citation) > 20

        results[domain] = {
            'citation': citation,
            'has_year': has_year,
            'has_author_ref': has_author,
            'has_instrument_name': has_instrument,
            'valid': has_year and has_author and has_instrument,
        }

        status = '✓' if results[domain]['valid'] else '✗'
        print(f"  {status} {domain:20s}: {citation[:80]}...")

    n_valid = sum(1 for v in results.values() if v['valid'])
    print(f"\n  {n_valid}/{len(results)} domains have valid citations")

    with open(output_dir / 'clinical_citations.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def validate_keyword_coverage(output_dir: Path, fig_dir: Path):
    """Analyze keyword overlap and coverage across domains."""
    print("\n" + "=" * 60)
    print("KEYWORD COVERAGE ANALYSIS")
    print("=" * 60)

    all_keywords = DEVELOPMAP.get_all_keywords()

    # Check for cross-domain keyword overlap
    overlaps = {}
    domain_names = list(all_keywords.keys())
    for i, d1 in enumerate(domain_names):
        for j, d2 in enumerate(domain_names):
            if i >= j:
                continue
            shared = all_keywords[d1] & all_keywords[d2]
            if shared:
                overlaps[f"{d1}_vs_{d2}"] = list(shared)

    results = {
        'per_domain_keyword_count': {d: len(kw) for d, kw in all_keywords.items()},
        'overlaps': overlaps,
        'total_unique_keywords': len(set().union(*all_keywords.values())),
    }

    print(f"\n  Keywords per domain:")
    for domain, count in sorted(results['per_domain_keyword_count'].items(),
                                 key=lambda x: x[1], reverse=True):
        print(f"    {domain:20s}: {count}")

    if overlaps:
        print(f"\n  Cross-domain overlaps:")
        for pair, keywords in overlaps.items():
            print(f"    {pair}: {keywords}")
    else:
        print(f"\n  No cross-domain keyword overlaps found")

    print(f"\n  Total unique keywords: {results['total_unique_keywords']}")

    with open(output_dir / 'keyword_coverage.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plot keyword counts
    fig, ax = plt.subplots(figsize=(10, 6))
    domains = list(results['per_domain_keyword_count'].keys())
    counts = [results['per_domain_keyword_count'][d] for d in domains]
    ax.barh(domains, counts, color='steelblue')
    ax.set_xlabel('Number of Keywords')
    ax.set_title('DevelopMap Keywords per Domain')
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(fig_dir / 'keyword_counts.png', dpi=150)
    plt.close(fig)

    return results


def validate_with_amazon_data(amazon_dir: str, output_dir: Path, fig_dir: Path):
    """Classify Amazon products and compute per-domain precision/recall."""
    print("\n" + "=" * 60)
    print("AMAZON PRODUCT CLASSIFICATION VALIDATION")
    print("=" * 60)

    from src.data.amazon_product_loader import AmazonProductLoader

    try:
        loader = AmazonProductLoader(amazon_dir, use_embeddings=False)
        report = loader.load_and_classify()
    except FileNotFoundError as e:
        print(f"\n  ⚠ Amazon data not available: {e}")
        print("  Skipping Amazon product validation.")
        return None

    # Save report
    loader.save_report(str(output_dir / 'amazon_classification_report.json'), report)

    # Plot domain distribution
    domain_counts = report.get('per_domain_counts', {})
    if domain_counts:
        # Filter out None
        domain_counts = {str(k): v for k, v in domain_counts.items() if k is not None}

        fig, ax = plt.subplots(figsize=(10, 6))
        domains = sorted(domain_counts.keys())
        counts = [domain_counts[d] for d in domains]
        ax.barh(domains, counts, color='coral')
        ax.set_xlabel('Number of Products')
        ax.set_title('Amazon Products per Developmental Domain')
        ax.grid(True, alpha=0.3, axis='x')
        fig.tight_layout()
        fig.savefig(fig_dir / 'amazon_domain_distribution.png', dpi=150)
        plt.close(fig)

    # Plot confidence distributions
    conf_dists = report.get('confidence_distributions', {})
    if conf_dists:
        fig, ax = plt.subplots(figsize=(10, 6))
        domains = sorted(conf_dists.keys())
        means = [conf_dists[d]['mean'] for d in domains]
        stds = [conf_dists[d]['std'] for d in domains]
        ax.barh(domains, means, xerr=stds, color='teal', capsize=3)
        ax.set_xlabel('Mean Classification Confidence')
        ax.set_title('Classification Confidence per Domain')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis='x')
        fig.tight_layout()
        fig.savefig(fig_dir / 'amazon_confidence.png', dpi=150)
        plt.close(fig)

    return report


def validate_with_synthetic_products(output_dir: Path, fig_dir: Path):
    """Use known synthetic product names as a sanity check."""
    print("\n" + "=" * 60)
    print("SYNTHETIC PRODUCT SANITY CHECK")
    print("=" * 60)

    classifier = ProductClassifier(use_embeddings=False)

    test_products = [
        ("Wooden alphabet puzzle for toddlers", "fine_motor"),
        ("Balance bike for kids", "gross_motor"),
        ("Board books for babies", "language"),
        ("Pretend play kitchen set", "social_emotional"),
        ("Fidget spinner sensory toy", "sensory"),
        ("Adaptive utensils for special needs", "adaptive"),
        ("White noise machine for baby sleep", "sleep"),
        ("Divided plate for picky eaters", "feeding"),
        ("Visual schedule for daily routine", "behavioral"),
        ("Speech therapy workbook", "therapeutic"),
        ("Crayons and coloring book", "fine_motor"),
        ("Soccer ball for outdoor play", "gross_motor"),
        ("Flashcards for learning words", "language"),
        ("Weighted blanket for calming", "sensory"),
        ("Reward chart with stickers", "behavioral"),
    ]

    correct = 0
    total = len(test_products)
    results = []

    for text, expected in test_products:
        primary = classifier.get_primary_domain(text)
        predicted = primary[0] if primary else 'unmapped'
        match = predicted == expected
        correct += int(match)
        status = '✓' if match else '✗'
        print(f"  {status} '{text[:40]}...' → {predicted} (expected: {expected})")
        results.append({
            'text': text,
            'expected': expected,
            'predicted': predicted,
            'correct': match,
        })

    accuracy = correct / total
    print(f"\n  Accuracy: {correct}/{total} = {accuracy:.1%}")

    with open(output_dir / 'synthetic_sanity_check.json', 'w') as f:
        json.dump({'accuracy': accuracy, 'results': results}, f, indent=2)

    return {'accuracy': accuracy, 'results': results}


def main():
    parser = argparse.ArgumentParser(description='Validate DevelopMap taxonomy')
    parser.add_argument('--amazon_dir', type=str, default='data/raw/amazon',
                        help='Path to Amazon product metadata directory')
    parser.add_argument('--output_dir', type=str, default='results/taxonomy',
                        help='Output directory for results')
    parser.add_argument('--fig_dir', type=str, default='figures/taxonomy',
                        help='Output directory for figures')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DEVELOPMAP TAXONOMY VALIDATION")
    print("=" * 60)

    # 1. ASQ-3 alignment
    asq3_results = validate_asq3_alignment(output_dir)

    # 2. Clinical citations
    citation_results = validate_clinical_citations(output_dir)

    # 3. Keyword coverage
    keyword_results = validate_keyword_coverage(output_dir, fig_dir)

    # 4. Synthetic product sanity check
    sanity_results = validate_with_synthetic_products(output_dir, fig_dir)

    # 5. Amazon product validation (if data available)
    amazon_results = validate_with_amazon_data(args.amazon_dir, output_dir, fig_dir)

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    n_citations_valid = sum(1 for v in citation_results.values() if v['valid'])
    n_asq3_mapped = sum(1 for v in asq3_results.values() if v['has_asq3_mapping'])
    print(f"  Clinical citations: {n_citations_valid}/{len(citation_results)} valid")
    print(f"  ASQ-3 mappings: {n_asq3_mapped}/{len(asq3_results)} domains")
    print(f"  Keyword overlap pairs: {len(keyword_results['overlaps'])}")
    print(f"  Sanity check accuracy: {sanity_results['accuracy']:.1%}")
    if amazon_results:
        print(f"  Amazon unmapped rate: {amazon_results['unmapped_rate']:.1%}")
    print(f"\n  Results saved to: {output_dir}/")
    print(f"  Figures saved to: {fig_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
