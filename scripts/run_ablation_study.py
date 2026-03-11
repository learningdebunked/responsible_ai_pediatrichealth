#!/usr/bin/env python
"""Ablation study for RetailHealth synthetic data pipeline.

Runs 5 experiments to quantify how much the model's performance depends
on the hardcoded purchase multipliers vs. learning genuine patterns:

1. Multiplier degradation: Scale delay multipliers from 1.0 (no signal)
   to current values. Plot AUROC vs multiplier strength.
2. Domain randomization: Shuffle which domains are elevated for which
   delay types. If AUROC stays the same, the model learns something trivial.
3. Noise injection: Add increasing random noise to domain counts.
   Plot AUROC vs noise level.
4. Baseline comparison: Train logistic regression on demographics only
   (no purchase data). Compare AUROC to Transformer.
5. Cross-seed validation: Generate 5 datasets with different seeds.
   Train and evaluate each. Report mean +/- std AUROC.

Usage:
    python scripts/run_ablation_study.py --n_families 5000 --output_dir results/ablation
"""

import argparse
import sys
import json
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic_generator import SyntheticDataGenerator
from src.data.preprocessing import TransactionPreprocessor


def _quick_train_evaluate(families_df, transactions_df, model_type='logistic'):
    """Quick train/evaluate cycle for ablation experiments.

    Uses logistic regression by default for speed; optionally uses a
    small Transformer for comparison.

    Returns:
        Dict with auroc, precision, recall, f1
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split

    # Build feature matrix: mean domain counts per family
    domain_names = transactions_df['domain'].unique().tolist()
    features = []
    labels = []

    for _, fam in families_df.iterrows():
        fam_txns = transactions_df[transactions_df['family_id'] == fam['family_id']]
        domain_counts = fam_txns['domain'].value_counts()
        row = [domain_counts.get(d, 0) for d in domain_names]
        features.append(row)
        labels.append(int(fam['has_delay']))

    X = np.array(features, dtype=np.float32)
    y = np.array(labels)

    if len(np.unique(y)) < 2:
        return {'auroc': 0.5, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if model_type == 'logistic':
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
    else:
        # Simple neural network via sklearn MLPClassifier as Transformer proxy
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                            random_state=42)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

    try:
        auroc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auroc = 0.5

    return {
        'auroc': float(auroc),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
    }


def _demographics_only_evaluate(families_df):
    """Train logistic regression on demographics only (no purchase data)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df = families_df.copy()
    le_geo = LabelEncoder()
    le_eth = LabelEncoder()
    df['geo_enc'] = le_geo.fit_transform(df['geography'].fillna('unknown'))
    df['eth_enc'] = le_eth.fit_transform(df['ethnicity'].fillna('unknown'))

    feature_cols = ['child_age_months', 'income_quintile', 'geo_enc', 'eth_enc']
    if 'n_children' in df.columns:
        feature_cols.append('n_children')

    X = df[feature_cols].values.astype(np.float32)
    y = df['has_delay'].astype(int).values

    if len(np.unique(y)) < 2:
        return {'auroc': 0.5}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]

    try:
        auroc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auroc = 0.5

    return {'auroc': float(auroc)}


# ── Experiment 1: Multiplier Degradation ─────────────────────────────────────

def experiment_multiplier_degradation(n_families, output_dir, fig_dir):
    """Scale delay multipliers from 1.0 to full strength."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Multiplier Degradation")
    print("=" * 60)

    scales = np.arange(0.0, 1.1, 0.1)
    results = []

    for scale in scales:
        print(f"  Scale={scale:.1f}...", end=" ", flush=True)
        gen = SyntheticDataGenerator(seed=42)

        # Monkey-patch the multiplier method to scale the signal
        original_method = gen._get_delay_purchase_multiplier

        def scaled_multiplier(delay_type, domain, months_since_onset, s=scale):
            orig = original_method(delay_type, domain, months_since_onset)
            return 1.0 + (orig - 1.0) * s

        gen._get_delay_purchase_multiplier = scaled_multiplier
        fam_df, txn_df = gen.generate_dataset(n_families=n_families, months_history=12)

        metrics = _quick_train_evaluate(fam_df, txn_df)
        metrics['scale'] = float(scale)
        results.append(metrics)
        print(f"AUROC={metrics['auroc']:.3f}")

    # Save results
    with open(output_dir / 'multiplier_degradation.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r['scale'] for r in results], [r['auroc'] for r in results],
            'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Multiplier Scale Factor')
    ax.set_ylabel('AUROC')
    ax.set_title('Model Performance vs. Multiplier Strength')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / 'multiplier_degradation.png', dpi=150)
    plt.close(fig)

    return results


# ── Experiment 2: Domain Randomization ────────────────────────────────────────

def experiment_domain_randomization(n_families, output_dir, fig_dir):
    """Randomly shuffle domain-delay mappings."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Domain Randomization")
    print("=" * 60)

    results = []

    # Original (non-randomized)
    gen = SyntheticDataGenerator(seed=42)
    fam_df, txn_df = gen.generate_dataset(n_families=n_families, months_history=12)
    orig_metrics = _quick_train_evaluate(fam_df, txn_df)
    orig_metrics['trial'] = 'original'
    results.append(orig_metrics)
    print(f"  Original: AUROC={orig_metrics['auroc']:.3f}")

    # 5 random shuffles
    for trial in range(5):
        gen = SyntheticDataGenerator(seed=100 + trial)

        # Shuffle domain assignments in multiplier table
        original_method = gen._get_delay_purchase_multiplier
        domain_names = gen.developmap.get_domain_names()
        shuffled = domain_names.copy()
        rng = np.random.RandomState(trial)
        rng.shuffle(shuffled)
        domain_map = dict(zip(domain_names, shuffled))

        def shuffled_multiplier(delay_type, domain, months_since_onset, dm=domain_map):
            mapped_domain = dm.get(domain, domain)
            return original_method(delay_type, mapped_domain, months_since_onset)

        gen._get_delay_purchase_multiplier = shuffled_multiplier
        fam_df, txn_df = gen.generate_dataset(n_families=n_families, months_history=12)

        metrics = _quick_train_evaluate(fam_df, txn_df)
        metrics['trial'] = f'shuffle_{trial}'
        results.append(metrics)
        print(f"  Shuffle {trial}: AUROC={metrics['auroc']:.3f}")

    with open(output_dir / 'domain_randomization.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r['trial'] for r in results]
    aurocs = [r['auroc'] for r in results]
    colors = ['green'] + ['steelblue'] * 5
    ax.bar(labels, aurocs, color=colors)
    ax.set_ylabel('AUROC')
    ax.set_title('AUROC: Original vs. Shuffled Domain Mappings')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=30, ha='right')
    fig.tight_layout()
    fig.savefig(fig_dir / 'domain_randomization.png', dpi=150)
    plt.close(fig)

    return results


# ── Experiment 3: Noise Injection ─────────────────────────────────────────────

def experiment_noise_injection(n_families, output_dir, fig_dir):
    """Add increasing random noise to domain counts."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Noise Injection")
    print("=" * 60)

    gen = SyntheticDataGenerator(seed=42)
    fam_df, txn_df = gen.generate_dataset(n_families=n_families, months_history=12)

    noise_levels = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    results = []

    for noise_std in noise_levels:
        print(f"  Noise std={noise_std:.1f}...", end=" ", flush=True)
        noisy_txn = txn_df.copy()

        if noise_std > 0:
            # Add random noise domain reassignments
            domain_names = gen.developmap.get_domain_names()
            n_noisy = int(len(noisy_txn) * noise_std / 10.0)
            n_noisy = min(n_noisy, len(noisy_txn))
            noise_idx = np.random.choice(len(noisy_txn), n_noisy, replace=False)
            for idx in noise_idx:
                noisy_txn.at[idx, 'domain'] = np.random.choice(domain_names)

        metrics = _quick_train_evaluate(fam_df, noisy_txn)
        metrics['noise_std'] = noise_std
        results.append(metrics)
        print(f"AUROC={metrics['auroc']:.3f}")

    with open(output_dir / 'noise_injection.json', 'w') as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r['noise_std'] for r in results], [r['auroc'] for r in results],
            'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Noise Level (fraction of reassigned domains)')
    ax.set_ylabel('AUROC')
    ax.set_title('Model Performance vs. Noise Level')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / 'noise_injection.png', dpi=150)
    plt.close(fig)

    return results


# ── Experiment 4: Baseline Comparison ─────────────────────────────────────────

def experiment_baseline_comparison(n_families, output_dir, fig_dir):
    """Compare demographics-only logistic regression vs. purchase-based model."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Baseline Comparison")
    print("=" * 60)

    gen = SyntheticDataGenerator(seed=42)
    fam_df, txn_df = gen.generate_dataset(n_families=n_families, months_history=12)

    # Demographics-only baseline
    demo_metrics = _demographics_only_evaluate(fam_df)
    print(f"  Demographics-only LR: AUROC={demo_metrics['auroc']:.3f}")

    # Purchase-based logistic regression
    purchase_lr = _quick_train_evaluate(fam_df, txn_df, model_type='logistic')
    print(f"  Purchase-based LR:    AUROC={purchase_lr['auroc']:.3f}")

    # Purchase-based neural network (Transformer proxy)
    purchase_nn = _quick_train_evaluate(fam_df, txn_df, model_type='mlp')
    print(f"  Purchase-based MLP:   AUROC={purchase_nn['auroc']:.3f}")

    results = {
        'demographics_only': demo_metrics,
        'purchase_logistic': purchase_lr,
        'purchase_mlp': purchase_nn,
        'auroc_gap_lr': purchase_lr['auroc'] - demo_metrics['auroc'],
        'auroc_gap_mlp': purchase_nn['auroc'] - demo_metrics['auroc'],
    }

    with open(output_dir / 'baseline_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    models = ['Demographics\nOnly (LR)', 'Purchase\nData (LR)', 'Purchase\nData (MLP)']
    aurocs = [demo_metrics['auroc'], purchase_lr['auroc'], purchase_nn['auroc']]
    colors = ['#d62728', '#2ca02c', '#1f77b4']
    ax.bar(models, aurocs, color=colors)
    ax.set_ylabel('AUROC')
    ax.set_title('Model AUROC: Demographics vs. Purchase Data')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(aurocs):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    fig.tight_layout()
    fig.savefig(fig_dir / 'baseline_comparison.png', dpi=150)
    plt.close(fig)

    return results


# ── Experiment 5: Cross-Seed Validation ───────────────────────────────────────

def experiment_cross_seed(n_families, output_dir, fig_dir):
    """Generate 5 datasets with different seeds, report mean +/- std AUROC."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Cross-Seed Validation")
    print("=" * 60)

    seeds = [42, 123, 456, 789, 1024]
    results = []

    for seed in seeds:
        print(f"  Seed={seed}...", end=" ", flush=True)
        gen = SyntheticDataGenerator(seed=seed)
        fam_df, txn_df = gen.generate_dataset(n_families=n_families, months_history=12)
        metrics = _quick_train_evaluate(fam_df, txn_df)
        metrics['seed'] = seed
        results.append(metrics)
        print(f"AUROC={metrics['auroc']:.3f}")

    aurocs = [r['auroc'] for r in results]
    summary = {
        'per_seed': results,
        'mean_auroc': float(np.mean(aurocs)),
        'std_auroc': float(np.std(aurocs)),
        'min_auroc': float(np.min(aurocs)),
        'max_auroc': float(np.max(aurocs)),
    }

    print(f"\n  Mean AUROC: {summary['mean_auroc']:.3f} +/- {summary['std_auroc']:.3f}")

    with open(output_dir / 'cross_seed.json', 'w') as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(s) for s in seeds], aurocs, color='steelblue')
    ax.axhline(y=summary['mean_auroc'], color='orange', linestyle='--',
               label=f"Mean={summary['mean_auroc']:.3f}")
    ax.fill_between(range(len(seeds)),
                     summary['mean_auroc'] - summary['std_auroc'],
                     summary['mean_auroc'] + summary['std_auroc'],
                     alpha=0.15, color='orange')
    ax.set_xlabel('Random Seed')
    ax.set_ylabel('AUROC')
    ax.set_title('Cross-Seed AUROC Stability')
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(fig_dir / 'cross_seed.png', dpi=150)
    plt.close(fig)

    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--n_families', type=int, default=5000,
                        help='Number of families per experiment')
    parser.add_argument('--output_dir', type=str, default='results/ablation',
                        help='Directory for results JSON files')
    parser.add_argument('--fig_dir', type=str, default='figures/ablation',
                        help='Directory for figures')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RETAILHEALTH ABLATION STUDY")
    print("=" * 60)
    print(f"Families per experiment: {args.n_families}")
    print(f"Results: {output_dir}/")
    print(f"Figures: {fig_dir}/")

    all_results = {}
    all_results['multiplier_degradation'] = experiment_multiplier_degradation(
        args.n_families, output_dir, fig_dir)
    all_results['domain_randomization'] = experiment_domain_randomization(
        args.n_families, output_dir, fig_dir)
    all_results['noise_injection'] = experiment_noise_injection(
        args.n_families, output_dir, fig_dir)
    all_results['baseline_comparison'] = experiment_baseline_comparison(
        args.n_families, output_dir, fig_dir)
    all_results['cross_seed'] = experiment_cross_seed(
        args.n_families, output_dir, fig_dir)

    # Save combined results
    with open(output_dir / 'all_ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("ABLATION STUDY COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}/")
    print(f"Figures saved to: {fig_dir}/")


if __name__ == '__main__':
    main()
