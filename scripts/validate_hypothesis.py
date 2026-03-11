#!/usr/bin/env python
"""Validate the core hypothesis: retail purchase patterns contain
detectable signals of developmental delays.

This script runs a structured series of tests to determine whether
the hypothesis holds under progressively harder conditions:

1. **Circularity test**: Train on synthetic data, measure AUROC,
   then quantify how much performance degrades when multiplier
   noise increases. If AUROC stays high even with heavy noise,
   the signal may be genuine; if it collapses, the "signal" was
   just the hardcoded multipliers.

2. **Prevalence calibration**: Compare synthetic delay prevalence
   to NSCH empirical rates and flag discrepancies.

3. **Demographic balance**: Check that delay rates don't correlate
   with demographics in ways that contradict NSCH data.

4. **Taxonomy coverage**: Verify DevelopMap domains map to ASQ-3
   screening domains and have proper clinical citations.

5. **Selection bias probe**: Estimate which populations are
   missing from the synthetic data and how that affects fairness.

Usage:
    python scripts/validate_hypothesis.py \
        --n_families 5000 \
        --output_dir results/hypothesis
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic_generator import SyntheticDataGenerator
from src.taxonomy.developmap import DEVELOPMAP
from src.evaluation.fairness import SelectionBiasAnalyzer


def test_circularity(n_families: int, output_dir: Path, fig_dir: Path) -> dict:
    """Test 1: How much signal survives when multiplier noise increases?"""
    print("\n" + "=" * 60)
    print("TEST 1: CIRCULARITY ANALYSIS")
    print("=" * 60)

    noise_levels = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    results = {'noise_levels': noise_levels, 'aurocs': []}

    for sigma in noise_levels:
        gen = SyntheticDataGenerator(seed=42)
        gen.multiplier_noise_sigma = sigma

        fam_df, txn_df = gen.generate_dataset(
            n_families=n_families, months_history=12
        )

        # Build feature matrix: domain purchase counts per family
        domains = DEVELOPMAP.get_domain_names()
        features = np.zeros((len(fam_df), len(domains)))

        for i, fid in enumerate(fam_df['family_id']):
            fam_txns = txn_df[txn_df['family_id'] == fid]
            for j, domain in enumerate(domains):
                features[i, j] = len(fam_txns[fam_txns['domain'] == domain])

        labels = fam_df['has_delay'].astype(int).values

        # Skip if too few positives
        if labels.sum() < 10 or (1 - labels).sum() < 10:
            results['aurocs'].append(0.5)
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(features)

        scores = cross_val_score(
            LogisticRegression(max_iter=500, class_weight='balanced'),
            X, labels, cv=5, scoring='roc_auc'
        )
        mean_auroc = float(np.mean(scores))
        results['aurocs'].append(mean_auroc)
        print(f"  sigma={sigma:.1f}: AUROC={mean_auroc:.3f} (+/- {np.std(scores):.3f})")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(noise_levels, results['aurocs'], 'o-', color='steelblue', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle='--', label='Random')
    ax.set_xlabel('Multiplier Noise Sigma')
    ax.set_ylabel('5-Fold AUROC')
    ax.set_title('Circularity Test: Signal vs. Noise')
    ax.set_ylim(0.4, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / 'circularity_test.png', dpi=150)
    plt.close(fig)

    # Interpretation
    if len(results['aurocs']) >= 2:
        drop = results['aurocs'][0] - results['aurocs'][-1]
        results['auroc_drop'] = float(drop)
        if drop > 0.15:
            results['interpretation'] = (
                "AUROC drops significantly with noise, suggesting the model "
                "relies heavily on hardcoded multipliers (circularity present)."
            )
        else:
            results['interpretation'] = (
                "AUROC is relatively stable across noise levels, suggesting "
                "some genuine signal beyond the hardcoded multipliers."
            )
    print(f"\n  {results.get('interpretation', 'N/A')}")

    return results


def test_prevalence_calibration(n_families: int, output_dir: Path) -> dict:
    """Test 2: Does synthetic prevalence match NSCH rates?"""
    print("\n" + "=" * 60)
    print("TEST 2: PREVALENCE CALIBRATION")
    print("=" * 60)

    gen = SyntheticDataGenerator(seed=42)
    fam_df, _ = gen.generate_dataset(n_families=n_families, months_history=3)

    # NSCH reference rates
    nsch_rates = {
        'any_delay': 0.178,
        'language': 0.05,
        'motor': 0.03,
        'asd': 0.027,
        'adhd': 0.097,
    }

    synthetic_rates = {}
    synthetic_rates['any_delay'] = float(fam_df['has_delay'].mean())

    for dtype in ['language', 'motor', 'asd', 'adhd']:
        mask = fam_df['delay_type'] == dtype
        synthetic_rates[dtype] = float(mask.mean())

    results = {'nsch_rates': nsch_rates, 'synthetic_rates': synthetic_rates, 'gaps': {}}

    print(f"\n  {'Condition':15s} {'NSCH':>8s} {'Synth':>8s} {'Gap':>8s}")
    print(f"  {'-'*41}")
    for key in nsch_rates:
        gap = abs(synthetic_rates.get(key, 0) - nsch_rates[key])
        results['gaps'][key] = float(gap)
        flag = ' !!!' if gap > 0.02 else ''
        print(f"  {key:15s} {nsch_rates[key]:8.3f} {synthetic_rates.get(key, 0):8.3f} {gap:8.3f}{flag}")

    return results


def test_demographic_balance(n_families: int, output_dir: Path) -> dict:
    """Test 3: Are delay rates balanced across demographics?"""
    print("\n" + "=" * 60)
    print("TEST 3: DEMOGRAPHIC BALANCE")
    print("=" * 60)

    gen = SyntheticDataGenerator(seed=42)
    fam_df, _ = gen.generate_dataset(n_families=n_families, months_history=3)

    results = {}

    for attr in ['income_quintile', 'geography', 'ethnicity']:
        if attr not in fam_df.columns:
            continue

        group_rates = fam_df.groupby(attr)['has_delay'].mean()
        results[attr] = {str(k): float(v) for k, v in group_rates.items()}

        max_gap = float(group_rates.max() - group_rates.min())
        results[f'{attr}_max_gap'] = max_gap

        print(f"\n  {attr}:")
        for group, rate in sorted(group_rates.items(), key=str):
            print(f"    {str(group):15s}: {rate:.3f}")
        print(f"    Max gap: {max_gap:.3f}")

        if max_gap > 0.05:
            print(f"    WARNING: Large delay rate gap across {attr}")

    return results


def test_taxonomy_coverage(output_dir: Path) -> dict:
    """Test 4: DevelopMap clinical alignment validation."""
    print("\n" + "=" * 60)
    print("TEST 4: TAXONOMY COVERAGE")
    print("=" * 60)

    asq3_results = DEVELOPMAP.validate_against_asq3()

    alignments = DEVELOPMAP.get_clinical_alignments()
    citation_valid = {}
    for domain, citation in alignments.items():
        has_year = any(c.isdigit() for c in citation)
        has_ref = '(' in citation and ')' in citation
        citation_valid[domain] = has_year and has_ref

    n_valid = sum(citation_valid.values())
    n_mapped = sum(1 for v in asq3_results.values() if v['has_asq3_mapping'])

    results = {
        'n_domains': len(alignments),
        'n_citations_valid': n_valid,
        'n_asq3_mapped': n_mapped,
        'citation_valid': citation_valid,
    }

    print(f"\n  Citations valid: {n_valid}/{len(alignments)}")
    print(f"  ASQ-3 mapped: {n_mapped}/{len(asq3_results)}")

    return results


def test_selection_bias(n_families: int, output_dir: Path) -> dict:
    """Test 5: Selection bias estimation."""
    print("\n" + "=" * 60)
    print("TEST 5: SELECTION BIAS PROBE")
    print("=" * 60)

    gen = SyntheticDataGenerator(seed=42)
    fam_df, _ = gen.generate_dataset(n_families=n_families, months_history=3)

    analyzer = SelectionBiasAnalyzer()

    report = analyzer.generate_selection_bias_report(fam_df)

    results = {
        'n_total': report['n_total'],
        'n_recommendations': len(report.get('recommendations', [])),
        'coverage_gaps': {},
    }

    for attr, gaps in report.get('coverage_gaps', {}).items():
        n_underrep = sum(1 for v in gaps.values() if v.get('underrepresented'))
        results['coverage_gaps'][attr] = {
            'n_groups': len(gaps),
            'n_underrepresented': n_underrep,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Validate core hypothesis: retail purchases signal developmental delays'
    )
    parser.add_argument('--n_families', type=int, default=5000)
    parser.add_argument('--output_dir', type=str, default='results/hypothesis')
    parser.add_argument('--fig_dir', type=str, default='figures/hypothesis')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HYPOTHESIS VALIDATION")
    print("Can retail purchase patterns detect developmental delays?")
    print("=" * 60)

    all_results = {}

    # Run all tests
    all_results['circularity'] = test_circularity(args.n_families, output_dir, fig_dir)
    all_results['prevalence'] = test_prevalence_calibration(args.n_families, output_dir)
    all_results['demographics'] = test_demographic_balance(args.n_families, output_dir)
    all_results['taxonomy'] = test_taxonomy_coverage(output_dir)
    all_results['selection_bias'] = test_selection_bias(args.n_families, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("HYPOTHESIS VALIDATION SUMMARY")
    print("=" * 60)

    circ = all_results['circularity']
    if circ.get('auroc_drop', 0) > 0.15:
        print("  [FAIL] Circularity: Signal depends heavily on hardcoded multipliers")
    else:
        print("  [PASS] Circularity: Some signal persists beyond multipliers")

    prev = all_results['prevalence']
    max_gap = max(prev['gaps'].values()) if prev['gaps'] else 0
    if max_gap > 0.02:
        print(f"  [WARN] Prevalence: Max gap vs NSCH = {max_gap:.3f}")
    else:
        print(f"  [PASS] Prevalence: All rates within 2% of NSCH")

    demo = all_results['demographics']
    demo_issues = [k for k in demo if k.endswith('_max_gap') and demo[k] > 0.05]
    if demo_issues:
        print(f"  [WARN] Demographics: Large gaps in {demo_issues}")
    else:
        print("  [PASS] Demographics: Delay rates balanced across groups")

    tax = all_results['taxonomy']
    if tax['n_citations_valid'] == tax['n_domains']:
        print(f"  [PASS] Taxonomy: All {tax['n_domains']} domains have valid citations")
    else:
        print(f"  [WARN] Taxonomy: {tax['n_citations_valid']}/{tax['n_domains']} valid")

    bias = all_results['selection_bias']
    if bias['n_recommendations'] > 0:
        print(f"  [WARN] Selection Bias: {bias['n_recommendations']} recommendations")
    else:
        print("  [PASS] Selection Bias: No coverage gaps detected")

    print(f"\n  Results saved to: {output_dir}/")
    print(f"  Figures saved to: {fig_dir}/")
    print("=" * 60)

    # Save
    with open(output_dir / 'hypothesis_validation.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == '__main__':
    main()
