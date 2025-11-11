#!/usr/bin/env python
"""Federated training script for RetailHealth.

Usage:
    python scripts/train_federated.py --data_dir data/synthetic --n_clients 10 --rounds 100
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data.preprocessing import TransactionPreprocessor, split_data
from src.federated.models import create_model
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer


def load_data(data_dir: str):
    """Load preprocessed data."""
    data_dir = Path(data_dir)
    
    families = pd.read_csv(data_dir / 'all_families.csv')
    transactions = pd.read_csv(data_dir / 'all_transactions.csv')
    
    # Preprocess
    preprocessor = TransactionPreprocessor(months_history=24)
    data = preprocessor.process_dataset(transactions, families, normalize='standard')
    
    # Split
    train_data, val_data, test_data = split_data(data, train_ratio=0.6, val_ratio=0.2)
    
    return train_data, val_data, test_data


def split_data_for_clients(data: dict, n_clients: int):
    """Split data across clients."""
    n_samples = len(data['sequences'])
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    client_data = []
    samples_per_client = n_samples // n_clients
    
    for i in range(n_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < n_clients - 1 else n_samples
        client_indices = indices[start_idx:end_idx]
        
        client_data.append({
            'sequences': data['sequences'][client_indices],
            'labels': data['labels'][client_indices],
            'family_ids': data['family_ids'][client_indices]
        })
    
    return client_data


def main():
    parser = argparse.ArgumentParser(description='Train federated RetailHealth model')
    parser.add_argument('--data_dir', type=str, default='data/synthetic',
                       help='Directory with data')
    parser.add_argument('--n_clients', type=int, default=10,
                       help='Number of federated clients')
    parser.add_argument('--rounds', type=int, default=100,
                       help='Number of training rounds')
    parser.add_argument('--local_epochs', type=int, default=5,
                       help='Local epochs per round')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--model_type', type=str, default='transformer',
                       choices=['transformer', 'gru', 'lstm'],
                       help='Model architecture')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Hidden layer size')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of layers')
    parser.add_argument('--epsilon', type=float, default=0.5,
                       help='Privacy budget')
    parser.add_argument('--use_dp', action='store_true',
                       help='Use differential privacy')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("="*80)
    print("RetailHealth Federated Training")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Clients: {args.n_clients}")
    print(f"  Rounds: {args.rounds}")
    print(f"  Model: {args.model_type}")
    print(f"  Privacy: {'Enabled' if args.use_dp else 'Disabled'} (ε={args.epsilon})")
    print(f"  Device: {args.device}")
    print()
    
    # Load data
    print("Loading data...")
    train_data, val_data, test_data = load_data(args.data_dir)
    print(f"  Train: {len(train_data['sequences'])} samples")
    print(f"  Val: {len(val_data['sequences'])} samples")
    print(f"  Test: {len(test_data['sequences'])} samples")
    
    # Split data for clients
    print(f"\nSplitting data across {args.n_clients} clients...")
    client_datasets = split_data_for_clients(train_data, args.n_clients)
    
    # Create model
    print(f"\nCreating {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        input_size=10,  # Number of domains
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=0.1
    )
    
    # Create server
    server = FederatedServer(
        model=model,
        aggregation='fedavg',
        epsilon=args.epsilon,
        delta=1e-5,
        sampling_rate=1.0  # All clients participate
    )
    
    # Create clients
    print(f"Creating {args.n_clients} clients...")
    clients = []
    for i, client_data in enumerate(client_datasets):
        client_model = create_model(
            model_type=args.model_type,
            input_size=10,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=0.1
        )
        client = FederatedClient(
            client_id=f"client_{i}",
            model=client_model,
            train_data=client_data,
            device=args.device,
            use_dp=args.use_dp,
            epsilon=args.epsilon
        )
        clients.append(client)
    
    # Training loop
    print(f"\nStarting federated training...")
    best_val_acc = 0.0
    
    for round_num in tqdm(range(args.rounds), desc="Training"):
        # Train round
        stats = server.train_round(
            clients=clients,
            epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        
        # Evaluate every 10 rounds
        if (round_num + 1) % 10 == 0:
            val_metrics = server.evaluate(val_data, device=args.device)
            
            print(f"\nRound {round_num + 1}/{args.rounds}:")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            if args.use_dp:
                print(f"  Privacy (ε): {stats['epsilon']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(server.model.state_dict(), 
                          output_dir / f"{args.model_type}_best.pt")
                print(f"  ✓ Saved best model (acc={best_val_acc:.4f})")
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)
    test_metrics = server.evaluate(test_data, device=args.device)
    print(f"\nTest Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    
    # Save final model
    output_dir = Path(args.output_dir)
    torch.save(server.model.state_dict(), 
              output_dir / f"{args.model_type}_final.pt")
    print(f"\n✓ Training complete! Models saved to {output_dir}")


if __name__ == '__main__':
    main()