#!/usr/bin/env python
"""Script to generate synthetic retail transaction data.

Usage:
    python scripts/generate_synthetic_data.py --n_families 100000 --output_dir data/synthetic
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data.synthetic_generator import SyntheticDataGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic retail transaction data for RetailHealth"
    )
    parser.add_argument(
        "--n_families",
        type=int,
        default=100000,
        help="Number of families to generate (default: 100000)"
    )
    parser.add_argument(
        "--months_history",
        type=int,
        default=24,
        help="Months of transaction history (default: 24)"
    )
    parser.add_argument(
        "--delay_prevalence",
        type=float,
        default=0.15,
        help="Overall delay prevalence (default: 0.15)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/synthetic",
        help="Output directory for generated data (default: data/synthetic)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.6,
        help="Training set proportion (default: 0.6)"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation set proportion (default: 0.2)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("RetailHealth Synthetic Data Generator")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Families: {args.n_families:,}")
    print(f"  History: {args.months_history} months")
    print(f"  Delay prevalence: {args.delay_prevalence:.1%}")
    print(f"  Output: {output_dir}")
    print(f"  Seed: {args.seed}")
    print(f"  Splits: {args.train_split:.0%} train / {args.val_split:.0%} val / {1-args.train_split-args.val_split:.0%} test")
    print()
    
    # Initialize generator
    generator = SyntheticDataGenerator(seed=args.seed)
    
    # Generate dataset
    families_df, transactions_df = generator.generate_dataset(
        n_families=args.n_families,
        months_history=args.months_history,
        delay_prevalence=args.delay_prevalence
    )
    
    # Split into train/val/test
    print(f"\nSplitting data...")
    n_families = len(families_df)
    n_train = int(n_families * args.train_split)
    n_val = int(n_families * args.val_split)
    
    # Shuffle families
    families_df = families_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    train_families = families_df.iloc[:n_train]
    val_families = families_df.iloc[n_train:n_train+n_val]
    test_families = families_df.iloc[n_train+n_val:]
    
    # Split transactions
    train_transactions = transactions_df[transactions_df['family_id'].isin(train_families['family_id'])]
    val_transactions = transactions_df[transactions_df['family_id'].isin(val_families['family_id'])]
    test_transactions = transactions_df[transactions_df['family_id'].isin(test_families['family_id'])]
    
    print(f"  Train: {len(train_families):,} families, {len(train_transactions):,} transactions")
    print(f"  Val: {len(val_families):,} families, {len(val_transactions):,} transactions")
    print(f"  Test: {len(test_families):,} families, {len(test_transactions):,} transactions")
    
    # Save datasets
    print(f"\nSaving datasets to {output_dir}...")
    
    # Save families
    train_families.to_csv(output_dir / "train_families.csv", index=False)
    val_families.to_csv(output_dir / "val_families.csv", index=False)
    test_families.to_csv(output_dir / "test_families.csv", index=False)
    
    # Save transactions
    train_transactions.to_csv(output_dir / "train_transactions.csv", index=False)
    val_transactions.to_csv(output_dir / "val_transactions.csv", index=False)
    test_transactions.to_csv(output_dir / "test_transactions.csv", index=False)
    
    # Save combined for convenience
    families_df.to_csv(output_dir / "all_families.csv", index=False)
    transactions_df.to_csv(output_dir / "all_transactions.csv", index=False)
    
    # Save metadata
    metadata = {
        'n_families': args.n_families,
        'months_history': args.months_history,
        'delay_prevalence': args.delay_prevalence,
        'seed': args.seed,
        'train_split': args.train_split,
        'val_split': args.val_split,
        'test_split': 1 - args.train_split - args.val_split,
        'n_transactions': len(transactions_df),
        'delay_counts': families_df[families_df['has_delay']]['delay_type'].value_counts().to_dict()
    }
    
    import json
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*80)
    print("✓ Data generation complete!")
    print("="*80)
    print(f"\nFiles saved:")
    for file in sorted(output_dir.glob("*.csv")):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {file.name}: {size_mb:.1f} MB")
    print(f"  metadata.json")
    print(f"\nNext steps:")
    print(f"  1. Train federated model: python scripts/train_federated.py --data_dir {output_dir}")
    print(f"  2. Evaluate model: python scripts/evaluate.py --test_data {output_dir}/test_families.csv")
    print()


if __name__ == "__main__":
    main()