"""Federated learning server."""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import numpy as np
from ..privacy.secure_aggregation import FederatedAveraging, FederatedProximal
from ..privacy.dp_mechanisms import RenyiDPAccountant


class FederatedServer:
    """FL server for coordinating training."""
    
    def __init__(self, model: nn.Module, aggregation: str = 'fedavg',
                 epsilon: float = 1.0, delta: float = 1e-5, sampling_rate: float = 0.1):
        """Initialize FL server.
        
        Args:
            model: Global model
            aggregation: Aggregation method ('fedavg' or 'fedprox')
            epsilon: Privacy budget
            delta: Privacy parameter
            sampling_rate: Client sampling rate
        """
        self.model = model
        self.aggregation_method = aggregation
        
        if aggregation == 'fedavg':
            self.aggregator = FederatedAveraging()
        elif aggregation == 'fedprox':
            self.aggregator = FederatedProximal(mu=0.01)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        self.accountant = RenyiDPAccountant(epsilon, delta, sampling_rate)
        self.round = 0
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get global model parameters."""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set global model parameters."""
        for name, param in self.model.named_parameters():
            param.data = parameters[name].clone()
    
    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates.
        
        Args:
            client_updates: List of (parameters, data_size) tuples
            
        Returns:
            Aggregated parameters
        """
        client_models = [update[0] for update in client_updates]
        client_sizes = [update[1] for update in client_updates]
        
        if self.aggregation_method == 'fedavg':
            aggregated = self.aggregator.aggregate(client_models, client_sizes)
        else:  # fedprox
            global_model = self.get_parameters()
            aggregated = self.aggregator.aggregate(client_models, client_sizes, global_model)
        
        return aggregated
    
    def train_round(self, clients: List, epochs: int, batch_size: int, lr: float) -> Dict:
        """Execute one training round.
        
        Args:
            clients: List of FederatedClient instances
            epochs: Local epochs
            batch_size: Batch size
            lr: Learning rate
            
        Returns:
            Round statistics
        """
        # Broadcast global model
        global_params = self.get_parameters()
        for client in clients:
            client.set_parameters(global_params)
        
        # Collect client updates
        client_updates = []
        for client in clients:
            params, size = client.train(epochs, batch_size, lr)
            client_updates.append((params, size))
        
        # Aggregate
        aggregated_params = self.aggregate(client_updates)
        self.set_parameters(aggregated_params)
        
        # Update privacy accounting
        self.accountant.step()
        self.round += 1
        
        return {
            'round': self.round,
            'num_clients': len(clients),
            'epsilon': self.accountant.get_current_epsilon(1.0)
        }
    
    def evaluate(self, test_data: Dict, device: str = 'cpu') -> Dict[str, float]:
        """Evaluate global model.
        
        Args:
            test_data: Test data dictionary
            device: Device to evaluate on
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        self.model.to(device)
        
        sequences = torch.FloatTensor(test_data['sequences']).to(device)
        labels = torch.FloatTensor(test_data['labels']).unsqueeze(1).to(device)
        
        with torch.no_grad():
            outputs = self.model(sequences)
            predictions = (outputs > 0.5).float()
            
            accuracy = (predictions == labels).float().mean().item()
            loss = nn.BCELoss()(outputs, labels).item()
        
        return {'accuracy': accuracy, 'loss': loss}