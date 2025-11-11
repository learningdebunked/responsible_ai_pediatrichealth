"""Temporal encoding for sequence data."""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalEncoder(nn.Module):
    """Encode temporal sequences with various architectures."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2,
                 encoder_type: str = 'gru', dropout: float = 0.1):
        """Initialize temporal encoder.
        
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            num_layers: Number of layers
            encoder_type: Type of encoder ('gru', 'lstm', 'transformer')
            dropout: Dropout rate
        """
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_size = hidden_size
        
        if encoder_type == 'gru':
            self.encoder = nn.GRU(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        elif encoder_type == 'lstm':
            self.encoder = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
        elif encoder_type == 'transformer':
            self.input_proj = nn.Linear(input_size, hidden_size)
            self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=8, dim_feedforward=hidden_size*4,
                dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Encode temporal sequence.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            mask: Optional attention mask
            
        Returns:
            Encoded representation
        """
        if self.encoder_type in ['gru', 'lstm']:
            output, hidden = self.encoder(x)
            # Return last hidden state
            if self.encoder_type == 'lstm':
                hidden = hidden[0]  # Take hidden state, not cell state
            return hidden[-1]  # Last layer
        
        elif self.encoder_type == 'transformer':
            x = self.input_proj(x)
            x = self.pos_encoder(x)
            output = self.encoder(x, src_key_padding_mask=mask)
            # Global average pooling
            return output.mean(dim=1)


class TemporalFeatureExtractor(nn.Module):
    """Extract temporal features from sequences."""
    
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract temporal features.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            
        Returns:
            Feature tensor with temporal statistics
        """
        # Basic statistics
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        max_val = x.max(dim=1)[0]
        min_val = x.min(dim=1)[0]
        
        # Trend (difference between first and last)
        trend = x[:, -1, :] - x[:, 0, :]
        
        # Concatenate all features
        features = torch.cat([mean, std, max_val, min_val, trend], dim=1)
        
        return features