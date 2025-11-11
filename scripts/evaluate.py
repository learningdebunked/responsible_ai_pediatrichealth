#!/usr/bin/env python
"""Evaluation script for RetailHealth models.

Usage:
    python scripts/evaluate.py --model_path models/transformer_best.pt --test_data data/synthetic
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
import numpy as np

from src.data.preprocessing import TransactionPreprocessor
from src.federated.models import create_model
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.fairness import FairnessAnalyzer, check_fairness_thresholds


def load_test_data(data_dir: str):
    """Load and preprocess test data."""
    data_dir = Path(data_dir)
    
    families = pd.read_csv(data_dir / 'test_families.csv')
    transactions = pd.read_csv(data_dir / 'test_transactions.csv')
    
    preprocessor = TransactionPreprocessor(months_history=24)
    data = preprocessor.process_dataset(transactions, families, normalize='standard')
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Evaluate RetailHealth model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data directory')
    parser.add_argument('--model_type', type=str, default='transformer',
                       choices=['transformer', 'gru', 'lstm'],
                       help='Model architecture')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of layers')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    parser.add_argument('--fairness_analysis', action='store_true',
                       help='Perform fairness analysis')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("RetailHealth Model Evaluation")
    print("="*80)
    print(f"\nModel: {args.model_path}")
    print(f"Test Data: {args.test_data}")
    print()
    
    # Load test data
    print("Loading test data...")
    test_data = load_test_data(args.test_data)
    print(f"  Test samples: {len(test_data['sequences'])}")
    print(f"  Positive samples: {test_data['labels'].sum()}")
    
    # Create model
    print(f"\nLoading {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        input_size=10,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=0.1
    )
    
    # Load weights
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.to(args.device)
    model.eval()
    
    # Get predictions
    print("\nGenerating predictions...")
    sequences = torch.FloatTensor(test_data['sequences']).to(args.device)
    
    with torch.no_grad():
        outputs = model(sequences)
        y_prob = outputs.cpu().numpy().flatten()
        y_pred = (y_prob > 0.5).astype(int)
    
    y_true = test_data['labels']
    
    # Basic evaluation
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    
    evaluator = ModelEvaluator(threshold=0.5)
    
    # Compute early detection lead time if available
    if 'child_age_months' in test_data['metadata'].columns and 'delay_onset_month' in test_data['metadata'].columns:
        detection_ages = test_data['metadata']['child_age_months'].values
        # Assume diagnosis happens 18 months after onset on average
        diagnosis_ages = test_data['metadata']['delay_onset_month'].values + 18
        diagnosis_ages = np.where(test_data['metadata']['has_delay'], diagnosis_ages, detection_ages)
        
        results = evaluator.evaluate_comprehensive(
            y_true, y_pred, y_prob,
            detection_ages=detection_ages,
            diagnosis_ages=diagnosis_ages
        )
    else:
        results = evaluator.evaluate_comprehensive(y_true, y_pred, y_prob)
    
    # Print metrics
    metrics = results['basic_metrics']
    print(f"\nOverall Performance:")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  FPR: {metrics['fpr']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives: {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  True Negatives: {metrics['true_negatives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    
    if 'lead_time' in results:
        lt = results['lead_time']
        print(f"\nEarly Detection Lead Time:")
        print(f"  Mean: {lt['mean_lead_time']:.2f} months")
        print(f"  Median: {lt['median_lead_time']:.2f} months")
        print(f"  Std Dev: {lt['std_lead_time']:.2f} months")
        print(f"  Early Detections: {lt['n_early_detections']}")
    
    # Fairness analysis
    if args.fairness_analysis:
        print("\n" + "="*80)
        print("FAIRNESS ANALYSIS")
        print("="*80)
        
        fairness_analyzer = FairnessAnalyzer(
            sensitive_attributes=['income_quintile', 'geography', 'ethnicity']
        )
        
        fairness_results = fairness_analyzer.analyze_all(
            y_true, y_pred, y_prob, test_data['metadata']
        )
        
        # Print fairness report
        report = fairness_analyzer.generate_report(fairness_results)
        print(report)
        
        # Check thresholds
        passes, violations = check_fairness_thresholds(
            fairness_results,
            tpr_gap_threshold=0.10,
            fpr_gap_threshold=0.10
        )
        
        if passes:
            print("\n✓ All fairness thresholds met!")
        else:
            print("\n⚠ Fairness violations detected:")
            for violation in violations:
                print(f"  - {violation}")
    
    # Performance by delay type
    if 'delay_type' in test_data['metadata'].columns:
        print("\n" + "="*80)
        print("PERFORMANCE BY DELAY TYPE")
        print("="*80)
        
        for delay_type in test_data['metadata']['delay_type'].dropna().unique():
            mask = test_data['metadata']['delay_type'] == delay_type
            n_samples = mask.sum()
            
            if n_samples > 0:
                type_metrics = evaluator.compute_basic_metrics(
                    y_true[mask], y_pred[mask], y_prob[mask]
                )
                
                print(f"\n{delay_type.upper()} (n={n_samples}):")
                
                # Only print AUROC if it's valid
                if not np.isnan(type_metrics['auroc']):
                    print(f"  AUROC: {type_metrics['auroc']:.4f}")
                else:
                    print(f"  AUROC: N/A (only one class present)")
                
                print(f"  Precision: {type_metrics['precision']:.4f}")
                print(f"  Recall: {type_metrics['recall']:.4f}")
                print(f"  F1: {type_metrics['f1']:.4f}")
                print(f"  Samples: TP={type_metrics['true_positives']}, "
                      f"FP={type_metrics['false_positives']}, "
                      f"TN={type_metrics['true_negatives']}, "
                      f"FN={type_metrics['false_negatives']}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    results_df = pd.DataFrame({
        'family_id': test_data['family_ids'],
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    })
    results_df.to_csv(output_dir / 'predictions.csv', index=False)
    
    # Save metrics
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({
            'basic_metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                            for k, v in metrics.items()}
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()