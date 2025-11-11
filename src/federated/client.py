"""Federated learning client."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple
from ..privacy.dp_mechanisms import DPMechanism


class FederatedClient:
    """FL client for local training."""
    
    def __init__(self, client_id: str, model: nn.Module, train_data: Dict, 
                 device: str = 'cpu', use_dp: bool = True, epsilon: float = 1.0):
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.use_dp = use_dp
        
        sequences = torch.FloatTensor(train_data['sequences'])
        labels = torch.FloatTensor(train_data['labels']).unsqueeze(1)
        self.dataset = TensorDataset(sequences, labels)
        self.data_size = len(self.dataset)
        
        if use_dp:
            self.dp_mechanism = DPMechanism(epsilon=epsilon, delta=1e-5)
        else:
            self.dp_mechanism = None
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters."""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set model parameters."""
        for name, param in self.model.named_parameters():
            param.data = parameters[name].clone().to(self.device)
    
    def train(self, epochs: int, batch_size: int, lr: float) -> Tuple[Dict[str, torch.Tensor], int]:
        """Train model locally."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for sequences, labels in dataloader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, labels)
                loss.backward()
                
                if self.use_dp and self.dp_mechanism:
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad = self.dp_mechanism.privatize_gradients(param.grad)
                
                optimizer.step()
        
        return self.get_parameters(), self.data_size