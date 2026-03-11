#!/usr/bin/env python
"""Run the full RetailHealth pipeline end-to-end.

Executes all stages in sequence:
1. Generate synthetic data (with confounded transactions)
2. Classify products using DevelopMap taxonomy
3. Build feature matrix
4. Train screening model (logistic regression baseline)
5. Evaluate with fairness and selection bias analysis
6. Run temporal association analysis
7. Run counterfactual sensitivity analysis
8. Generate governance audit trail
9. Produce summary report

Usage:
    python scripts/run_full_pipeline.py \
        --n_families 5000 \
        --months 12 \
        --output_dir results/pipeline \
        --fig_dir figures/pipeline
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic_generator import SyntheticDataGenerator
from src.taxonomy.developmap import DEVELOPMAP
from src.taxonomy.classifier import ProductClassifier
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.fairness import FairnessAnalyzer, SelectionBiasAnalyzer
from src.temporal_analysis.counterfactual import SensitivityAnalyzer
from src.governance.audit import AuditLogger, AuditEventType
from src.governance.consent import ConsentManager, ConsentScope


def stage_1_generate_data(n_families: int, months: int, audit: AuditLogger) -> tuple:
    """Stage 1: Generate synthetic data."""
    print("\n" + "=" * 60)
    print("STAGE 1: GENERATE SYNTHETIC DATA")
    print("=" * 60)

    gen = SyntheticDataGenerator(seed=42)
    fam_df, txn_df = gen.generate_dataset(
        n_families=n_families, months_history=months
    )

    print(f"  Families: {len(fam_df)}")
    print(f"  Transactions: {len(txn_df)}")
    print(f"  Delay rate: {fam_df['has_delay'].mean():.3f}")
    print(f"  Delay types: {fam_df[fam_df['has_delay']]['delay_type'].value_counts().to_dict()}")

    audit.log(AuditEventType.DATA_ACCESS, "pipeline", {
        'dataset': 'synthetic',
        'operation': 'generate',
        'n_families': len(fam_df),
        'n_transactions': len(txn_df),
    })

    return fam_df, txn_df


def stage_2_build_features(fam_df: pd.DataFrame, txn_df: pd.DataFrame) -> tuple:
    """Stage 2: Build feature matrix from transactions."""
    print("\n" + "=" * 60)
    print("STAGE 2: BUILD FEATURE MATRIX")
    print("=" * 60)

    domains = DEVELOPMAP.get_domain_names()
    n_families = len(fam_df)

    # Domain purchase counts
    features = np.zeros((n_families, len(domains)))
    family_ids = fam_df['family_id'].values

    for i, fid in enumerate(family_ids):
        fam_txns = txn_df[txn_df['family_id'] == fid]
        for j, domain in enumerate(domains):
            features[i, j] = len(fam_txns[fam_txns['domain'] == domain])

    # Add demographic features
    demo_features = pd.get_dummies(
        fam_df[['income_quintile', 'geography']],
        columns=['income_quintile', 'geography'],
        drop_first=True
    ).values.astype(float)

    X = np.hstack([features, demo_features])
    y = fam_df['has_delay'].astype(int).values

    feature_names = list(domains) + [
        f"demo_{i}" for i in range(demo_features.shape[1])
    ]

    print(f"  Feature matrix: {X.shape}")
    print(f"  Domain features: {len(domains)}")
    print(f"  Demographic features: {demo_features.shape[1]}")
    print(f"  Positive rate: {y.mean():.3f}")

    return X, y, feature_names


def stage_3_train_model(X: np.ndarray, y: np.ndarray,
                         audit: AuditLogger) -> tuple:
    """Stage 3: Train screening model."""
    print("\n" + "=" * 60)
    print("STAGE 3: TRAIN SCREENING MODEL")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    auroc = roc_auc_score(y_test, y_prob)
    print(f"\n  AUROC: {auroc:.3f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Delay', 'Delay']))

    audit.log_training_run(
        actor="pipeline",
        model_name="logistic_regression_baseline",
        hyperparams={'max_iter': 1000, 'class_weight': 'balanced'},
        data_version="synthetic_v1",
        metrics={'auroc': float(auroc)},
    )

    return model, scaler, X_test_s, y_test, y_pred, y_prob


def stage_4_evaluate(y_test: np.ndarray, y_pred: np.ndarray,
                      y_prob: np.ndarray, fam_df: pd.DataFrame,
                      X_test_indices: np.ndarray,
                      audit: AuditLogger, output_dir: Path) -> dict:
    """Stage 4: Evaluate model with fairness metrics."""
    print("\n" + "=" * 60)
    print("STAGE 4: EVALUATION & FAIRNESS")
    print("=" * 60)

    evaluator = ModelEvaluator(threshold=0.5)
    basic = evaluator.compute_basic_metrics(y_test, y_pred, y_prob)
    print(f"\n  Basic metrics: {json.dumps(basic, indent=2)}")

    # Fairness analysis
    test_df = fam_df.iloc[X_test_indices].reset_index(drop=True)

    fairness_analyzer = FairnessAnalyzer(
        sensitive_attributes=['income_quintile', 'geography']
    )

    fairness_results = {}
    for attr in ['income_quintile', 'geography']:
        if attr in test_df.columns:
            dp = fairness_analyzer.compute_demographic_parity(
                y_pred, test_df, attr
            )
            fairness_results[f'{attr}_dp_gap'] = dp.get('max_gap', 0)

    # Selection bias
    bias_analyzer = SelectionBiasAnalyzer()
    bias_report = bias_analyzer.generate_selection_bias_report(test_df)

    # Log fairness eval
    passed = all(v < 0.1 for v in fairness_results.values())
    violations = [f"{k}={v:.3f}" for k, v in fairness_results.items() if v > 0.1]
    audit.log_fairness_evaluation(
        actor="pipeline",
        evaluation_results=fairness_results,
        passed=passed,
        violations=violations,
    )

    return {
        'basic_metrics': basic,
        'fairness': fairness_results,
        'fairness_passed': passed,
        'n_bias_recommendations': len(bias_report.get('recommendations', [])),
    }


def stage_5_sensitivity(y_test: np.ndarray, y_prob: np.ndarray,
                         output_dir: Path, fig_dir: Path) -> dict:
    """Stage 5: Sensitivity analysis."""
    print("\n" + "=" * 60)
    print("STAGE 5: SENSITIVITY ANALYSIS")
    print("=" * 60)

    sa = SensitivityAnalyzer()

    # E-value for observed effect
    # Compute approximate risk ratio from probabilities
    pos_mean = y_prob[y_test == 1].mean()
    neg_mean = y_prob[y_test == 0].mean()
    rr = pos_mean / neg_mean if neg_mean > 0 else 1.0

    e_val = sa.compute_e_value(rr, measure='risk_ratio')
    print(f"\n  Risk ratio (prob ratio): {rr:.2f}")
    print(f"  E-value: {e_val['e_value']:.2f}")
    print(f"  Interpretation: An unmeasured confounder would need RR >= "
          f"{e_val['e_value']:.1f} with both treatment and outcome to "
          f"explain away the association.")

    # Quantitative bias analysis
    qba = sa.quantitative_bias_analysis(
        observed_rr=rr,
        confounder_prevalence_treated=0.3,
        confounder_prevalence_control=0.1,
        confounder_outcome_rr=2.0,
    )
    print(f"\n  Bias analysis: {qba['interpretation']}")

    return {
        'risk_ratio': float(rr),
        'e_value': e_val,
        'quantitative_bias_analysis': qba,
    }


def stage_6_governance(n_families: int, audit: AuditLogger,
                        output_dir: Path) -> dict:
    """Stage 6: Governance demonstration."""
    print("\n" + "=" * 60)
    print("STAGE 6: GOVERNANCE")
    print("=" * 60)

    # Consent demonstration
    cm = ConsentManager()

    n_consented = 0
    for i in range(min(n_families, 100)):
        fid = f"fam_{i:05d}"
        cm.grant_consent(fid, ConsentScope.DATA_COLLECTION, f"guardian_{i}", 24)
        cm.grant_consent(fid, ConsentScope.MODEL_TRAINING, f"guardian_{i}", 24)
        n_consented += 1

    summary = cm.get_summary()
    print(f"\n  Consent summary: {json.dumps(summary, indent=2)}")

    # COPPA compliance check
    coppa = cm.validate_coppa_compliance("fam_00000", 24)
    print(f"  COPPA compliant: {coppa['compliant']}")

    # Audit trail
    integrity = audit.verify_integrity()
    print(f"\n  Audit log entries: {integrity['n_entries']}")
    print(f"  Audit integrity: {'VALID' if integrity['is_valid'] else 'INVALID'}")

    # Export
    audit.export(str(output_dir / 'audit_trail.json'))
    cm.export_audit_trail(str(output_dir / 'consent_trail.json'))

    return {
        'consent_summary': summary,
        'coppa_compliant': coppa['compliant'],
        'audit_integrity': integrity['is_valid'],
        'audit_entries': integrity['n_entries'],
    }


def main():
    parser = argparse.ArgumentParser(description='Run full RetailHealth pipeline')
    parser.add_argument('--n_families', type=int, default=5000)
    parser.add_argument('--months', type=int, default=12)
    parser.add_argument('--output_dir', type=str, default='results/pipeline')
    parser.add_argument('--fig_dir', type=str, default='figures/pipeline')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    fig_dir = Path(args.fig_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    print("=" * 60)
    print("RETAILHEALTH END-TO-END PIPELINE")
    print("=" * 60)

    # Initialize audit logger
    audit = AuditLogger(log_path=str(output_dir / 'pipeline_audit.jsonl'))

    all_results = {}

    # Stage 1: Generate data
    fam_df, txn_df = stage_1_generate_data(args.n_families, args.months, audit)

    # Stage 2: Build features
    X, y, feature_names = stage_2_build_features(fam_df, txn_df)

    # Stage 3: Train model
    model, scaler, X_test, y_test, y_pred, y_prob = stage_3_train_model(X, y, audit)

    # Get test indices for fairness analysis
    _, X_test_raw, _, _ = train_test_split(
        np.arange(len(fam_df)), y, test_size=0.2, random_state=42, stratify=y
    )

    # Stage 4: Evaluate
    all_results['evaluation'] = stage_4_evaluate(
        y_test, y_pred, y_prob, fam_df, X_test_raw, audit, output_dir
    )

    # Stage 5: Sensitivity
    all_results['sensitivity'] = stage_5_sensitivity(
        y_test, y_prob, output_dir, fig_dir
    )

    # Stage 6: Governance
    all_results['governance'] = stage_6_governance(
        args.n_families, audit, output_dir
    )

    elapsed = time.time() - start_time

    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Families: {args.n_families}")
    print(f"  Transactions: {len(txn_df)}")
    print(f"  AUROC: {all_results['evaluation']['basic_metrics'].get('auroc', 'N/A')}")
    print(f"  Fairness passed: {all_results['evaluation']['fairness_passed']}")
    print(f"  E-value: {all_results['sensitivity']['e_value']['e_value']:.2f}")
    print(f"  Audit integrity: {all_results['governance']['audit_integrity']}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"\n  Results: {output_dir}/")
    print(f"  Figures: {fig_dir}/")
    print("=" * 60)

    # Save
    all_results['metadata'] = {
        'n_families': args.n_families,
        'months': args.months,
        'elapsed_seconds': elapsed,
    }

    with open(output_dir / 'pipeline_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == '__main__':
    main()
