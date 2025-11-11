"""Data preprocessing for RetailHealth."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from ..taxonomy.developmap import DEVELOPMAP


class TransactionPreprocessor:
    """Preprocess transaction data for model training."""
    
    def __init__(self, months_history: int = 24):
        """Initialize preprocessor.
        
        Args:
            months_history: Number of months of history to use
        """
        self.months_history = months_history
        self.domain_names = DEVELOPMAP.get_domain_names()
    
    def aggregate_by_month(self, transactions: pd.DataFrame, 
                          families: pd.DataFrame) -> pd.DataFrame:
        """Aggregate transactions by month and domain.
        
        Args:
            transactions: Transaction DataFrame
            families: Family DataFrame
            
        Returns:
            Aggregated DataFrame with monthly domain counts
        """
        # Ensure date column is datetime
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        
        # Add month column
        transactions['month'] = transactions['transaction_date'].dt.to_period('M')
        
        # Group by family, month, and domain
        monthly_counts = transactions.groupby(
            ['family_id', 'month', 'domain']
        ).size().reset_index(name='count')
        
        # Pivot to wide format
        wide_format = monthly_counts.pivot_table(
            index=['family_id', 'month'],
            columns='domain',
            values='count',
            fill_value=0
        ).reset_index()
        
        # Merge with family info
        result = wide_format.merge(families, on='family_id', how='left')
        
        return result
    
    def create_sequences(self, aggregated_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Create temporal sequences for each family.
        
        Args:
            aggregated_data: Aggregated monthly data
            
        Returns:
            Dictionary with sequences and labels
        """
        sequences = []
        labels = []
        family_ids = []
        
        for family_id in aggregated_data['family_id'].unique():
            family_data = aggregated_data[aggregated_data['family_id'] == family_id]
            
            # Sort by month
            family_data = family_data.sort_values('month')
            
            # Extract domain counts
            domain_counts = family_data[self.domain_names].values
            
            # Pad or truncate to fixed length
            if len(domain_counts) < self.months_history:
                # Pad with zeros at the beginning
                padding = np.zeros((self.months_history - len(domain_counts), len(self.domain_names)))
                domain_counts = np.vstack([padding, domain_counts])
            else:
                # Take last months_history months
                domain_counts = domain_counts[-self.months_history:]
            
            sequences.append(domain_counts)
            
            # Get label (has_delay)
            label = family_data['has_delay'].iloc[0]
            labels.append(int(label))
            family_ids.append(family_id)
        
        return {
            'sequences': np.array(sequences, dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64),
            'family_ids': np.array(family_ids)
        }
    
    def normalize_features(self, sequences: np.ndarray, 
                          method: str = 'standard') -> Tuple[np.ndarray, Dict]:
        """Normalize feature sequences.
        
        Args:
            sequences: Input sequences (n_families, months, domains)
            method: Normalization method ('standard', 'minmax', 'none')
            
        Returns:
            Tuple of (normalized_sequences, normalization_params)
        """
        if method == 'none':
            return sequences, {}
        
        # Compute statistics across all families and months
        flat_sequences = sequences.reshape(-1, sequences.shape[-1])
        
        if method == 'standard':
            mean = np.mean(flat_sequences, axis=0)
            std = np.std(flat_sequences, axis=0) + 1e-6
            normalized = (sequences - mean) / std
            params = {'mean': mean, 'std': std, 'method': 'standard'}
        
        elif method == 'minmax':
            min_val = np.min(flat_sequences, axis=0)
            max_val = np.max(flat_sequences, axis=0)
            range_val = max_val - min_val + 1e-6
            normalized = (sequences - min_val) / range_val
            params = {'min': min_val, 'max': max_val, 'method': 'minmax'}
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized, params
    
    def process_dataset(self, transactions: pd.DataFrame, 
                       families: pd.DataFrame,
                       normalize: str = 'standard') -> Dict:
        """Complete preprocessing pipeline.
        
        Args:
            transactions: Transaction DataFrame
            families: Family DataFrame
            normalize: Normalization method
            
        Returns:
            Dictionary with processed data
        """
        # Aggregate by month
        aggregated = self.aggregate_by_month(transactions, families)
        
        # Create sequences
        data = self.create_sequences(aggregated)
        
        # Normalize
        data['sequences'], data['norm_params'] = self.normalize_features(
            data['sequences'], method=normalize
        )
        
        # Add metadata
        family_metadata = families.set_index('family_id').loc[data['family_ids']]
        data['metadata'] = family_metadata.reset_index()
        
        return data


def split_data(data: Dict, train_ratio: float = 0.6, 
               val_ratio: float = 0.2, seed: int = 42) -> Tuple[Dict, Dict, Dict]:
    """Split data into train/val/test sets.
    
    Args:
        data: Processed data dictionary
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    n_samples = len(data['sequences'])
    indices = np.arange(n_samples)
    
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    def subset_data(idx):
        return {
            'sequences': data['sequences'][idx],
            'labels': data['labels'][idx],
            'family_ids': data['family_ids'][idx],
            'metadata': data['metadata'].iloc[idx].reset_index(drop=True),
            'norm_params': data.get('norm_params', {})
        }
    
    return subset_data(train_idx), subset_data(val_idx), subset_data(test_idx)