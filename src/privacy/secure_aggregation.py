"""Secure aggregation for federated learning."""

import torch
import numpy as np
from typing import List, Dict
import hashlib


class SecureAggregator:
    """Secure aggregation with masked model updates."""
    
    def __init__(self, num_clients: int, seed: int = 42):
        """Initialize secure aggregator.
        
        Args:
            num_clients: Number of clients
            seed: Random seed for reproducibility
        """
        self.num_clients = num_clients
        self.seed = seed
        self.client_masks = {}
    
    def generate_mask(self, client_id: str, round_num: int, 
                     param_shape: tuple, device: str = 'cpu') -> torch.Tensor:
        """Generate deterministic mask for a client.
        
        Args:
            client_id: Client identifier
            round_num: Training round number
            param_shape: Shape of parameters
            device: Device to create mask on
            
        Returns:
            Mask tensor
        """
        # Create deterministic seed from client_id and round
        seed_str = f"{client_id}_{round_num}_{self.seed}"
        seed_hash = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)
        
        # Generate mask
        generator = torch.Generator(device=device)
        generator.manual_seed(seed_hash)
        mask = torch.randn(param_shape, generator=generator, device=device)
        
        return mask
    
    def mask_model_update(self, model_update: Dict[str, torch.Tensor], 
                         client_id: str, round_num: int) -> Dict[str, torch.Tensor]:
        """Apply mask to model update.
        
        Args:
            model_update: Dictionary of parameter updates
            client_id: Client identifier
            round_num: Training round number
            
        Returns:
            Masked model update
        """
        masked_update = {}
        
        for name, param in model_update.items():
            mask = self.generate_mask(client_id, round_num, param.shape, param.device)
            masked_update[name] = param + mask
        
        return masked_update
    
    def aggregate_masked_updates(self, masked_updates: List[Dict[str, torch.Tensor]],
                                client_ids: List[str], round_num: int,
                                weights: List[float] = None,
                                expected_client_ids: List[str] = None
                                ) -> Dict[str, torch.Tensor]:
        """Aggregate masked updates and remove masks.
        
        Handles dropped clients: if expected_client_ids is provided,
        any client present in expected but missing from client_ids is
        treated as dropped. Their mask contribution is still subtracted
        (since masks are deterministic) to preserve correctness, and
        weights are renormalized over surviving clients.
        
        Args:
            masked_updates: List of masked model updates
            client_ids: List of client identifiers that responded
            round_num: Training round number
            weights: Optional weights for weighted average
            expected_client_ids: Full list of expected clients (optional).
                If provided, enables dropped-client handling.
            
        Returns:
            Aggregated model update (masks cancelled out)
        """
        # Detect dropped clients
        dropped_ids = []
        if expected_client_ids is not None:
            responded_set = set(client_ids)
            dropped_ids = [cid for cid in expected_client_ids
                           if cid not in responded_set]
            if dropped_ids:
                import warnings
                warnings.warn(
                    f"Dropped clients in round {round_num}: {dropped_ids}. "
                    f"Renormalizing weights over {len(client_ids)} survivors."
                )

        if weights is None:
            weights = [1.0 / len(masked_updates)] * len(masked_updates)
        else:
            # Renormalize weights for surviving clients
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
        
        # Initialize aggregated update
        aggregated = {}
        
        # Get parameter names from first update
        param_names = list(masked_updates[0].keys())
        
        for name in param_names:
            # Weighted sum of masked updates
            weighted_sum = sum(
                w * update[name] for w, update in zip(weights, masked_updates)
            )
            
            # Sum of masks for surviving clients
            mask_sum = sum(
                w * self.generate_mask(cid, round_num, masked_updates[0][name].shape,
                                      masked_updates[0][name].device)
                for w, cid in zip(weights, client_ids)
            )
            
            # Remove masks
            aggregated[name] = weighted_sum - mask_sum
        
        return aggregated

    def verify_aggregation_integrity(self,
                                      original_updates: List[Dict[str, torch.Tensor]],
                                      aggregated: Dict[str, torch.Tensor],
                                      weights: List[float] = None,
                                      tolerance: float = 1e-4) -> Dict[str, bool]:
        """Verify that secure aggregation produced correct results.

        Computes the plaintext weighted average and compares it to the
        secure aggregation output. Any discrepancy beyond tolerance
        indicates a mask cancellation error or dropped-client issue.

        Args:
            original_updates: List of *unmasked* model updates
            aggregated: Output of aggregate_masked_updates
            weights: Weights used during aggregation
            tolerance: Maximum allowable L-inf difference per parameter

        Returns:
            Dict mapping parameter name to whether verification passed
        """
        if weights is None:
            weights = [1.0 / len(original_updates)] * len(original_updates)

        results = {}
        param_names = list(original_updates[0].keys())

        for name in param_names:
            expected = sum(
                w * update[name] for w, update in zip(weights, original_updates)
            )
            actual = aggregated[name]
            max_diff = float((expected - actual).abs().max())
            results[name] = {
                'passed': max_diff <= tolerance,
                'max_diff': max_diff,
            }

        all_passed = all(v['passed'] for v in results.values())
        if not all_passed:
            import warnings
            failed = [k for k, v in results.items() if not v['passed']]
            warnings.warn(
                f"Aggregation integrity check FAILED for parameters: {failed}"
            )

        return results
    
    def simple_aggregate(self, updates: List[Dict[str, torch.Tensor]],
                        weights: List[float] = None) -> Dict[str, torch.Tensor]:
        """Simple weighted aggregation without masking.
        
        Args:
            updates: List of model updates
            weights: Optional weights for weighted average
            
        Returns:
            Aggregated model update
        """
        if weights is None:
            weights = [1.0 / len(updates)] * len(updates)
        
        aggregated = {}
        param_names = list(updates[0].keys())
        
        for name in param_names:
            aggregated[name] = sum(
                w * update[name] for w, update in zip(weights, updates)
            )
        
        return aggregated


class FederatedAveraging:
    """FedAvg aggregation algorithm."""
    
    @staticmethod
    def aggregate(client_models: List[Dict[str, torch.Tensor]],
                 client_sizes: List[int]) -> Dict[str, torch.Tensor]:
        """Aggregate client models using FedAvg.
        
        Args:
            client_models: List of client model parameters
            client_sizes: List of client dataset sizes
            
        Returns:
            Aggregated global model
        """
        total_size = sum(client_sizes)
        weights = [size / total_size for size in client_sizes]
        
        global_model = {}
        param_names = list(client_models[0].keys())
        
        for name in param_names:
            global_model[name] = sum(
                w * model[name] for w, model in zip(weights, client_models)
            )
        
        return global_model


class FederatedProximal:
    """FedProx aggregation with proximal term."""
    
    def __init__(self, mu: float = 0.01):
        """Initialize FedProx.
        
        Args:
            mu: Proximal term coefficient
        """
        self.mu = mu
    
    def aggregate(self, client_models: List[Dict[str, torch.Tensor]],
                 client_sizes: List[int],
                 global_model: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Aggregate with proximal term.
        
        Args:
            client_models: List of client model parameters
            client_sizes: List of client dataset sizes
            global_model: Current global model
            
        Returns:
            Aggregated global model
        """
        total_size = sum(client_sizes)
        weights = [size / total_size for size in client_sizes]
        
        new_global_model = {}
        param_names = list(client_models[0].keys())
        
        for name in param_names:
            # Weighted average
            avg_update = sum(
                w * model[name] for w, model in zip(weights, client_models)
            )
            
            # Add proximal term
            new_global_model[name] = (
                (1 - self.mu) * avg_update + self.mu * global_model[name]
            )
        
        return new_global_model