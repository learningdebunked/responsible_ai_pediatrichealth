"""Neural network models for developmental screening."""

import torch
import torch.nn as nn
from ..features.temporal_encoder import TemporalEncoder, PositionalEncoding


class DevelopmentalScreeningModel(nn.Module):
    """Base model for developmental screening."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 encoder_type: str = 'transformer', dropout: float = 0.1):
        """Initialize screening model.
        
        Args:
            input_size: Number of input features (domains)
            hidden_size: Hidden layer size
            num_layers: Number of encoder layers
            encoder_type: Type of encoder ('transformer', 'gru', 'lstm')
            dropout: Dropout rate
        """
        super().__init__()
        
        self.encoder = TemporalEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            encoder_type=encoder_type,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            mask: Optional attention mask
            
        Returns:
            Risk scores (batch, 1)
        """
        # Encode sequence
        encoded = self.encoder(x, mask)
        
        # Classify
        output = self.classifier(encoded)
        
        return output


class TransformerScreeningModel(nn.Module):
    """Transformer-based screening model."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 128,
                 num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input (batch, seq_len, input_size)
            
        Returns:
            Risk scores (batch, 1)
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)


class GRUScreeningModel(nn.Module):
    """GRU-based screening model."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input (batch, seq_len, input_size)
            
        Returns:
            Risk scores (batch, 1)
        """
        _, hidden = self.gru(x)
        x = hidden[-1]  # Last layer hidden state
        return self.classifier(x)


class LSTMScreeningModel(nn.Module):
    """LSTM-based screening model."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input (batch, seq_len, input_size)
            
        Returns:
            Risk scores (batch, 1)
        """
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]  # Last layer hidden state
        return self.classifier(x)


def create_model(model_type: str, input_size: int = 10, hidden_size: int = 128,
                num_layers: int = 4, **kwargs) -> nn.Module:
    """Factory function to create models.
    
    Args:
        model_type: Type of model ('transformer', 'gru', 'lstm')
        input_size: Input feature size
        hidden_size: Hidden layer size
        num_layers: Number of layers
        **kwargs: Additional model-specific arguments
        
    Returns:
        Model instance
    """
    if model_type == 'transformer':
        return TransformerScreeningModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            **kwargs
        )
    elif model_type == 'gru':
        return GRUScreeningModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            **kwargs
        )
    elif model_type == 'lstm':
        return LSTMScreeningModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")