"""Differential Privacy mechanisms."""

import torch
import math
from typing import Tuple


class DPMechanism:
    """DP mechanism with gradient clipping and noise."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, clip_norm: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.noise_multiplier = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    
    def clip_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """Clip gradients to bound sensitivity."""
        grad_norm = torch.norm(gradients, p=2)
        clip_factor = min(1.0, self.clip_norm / (grad_norm + 1e-6))
        return gradients * clip_factor
    
    def add_noise(self, gradients: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to gradients."""
        noise_std = self.noise_multiplier * self.clip_norm
        noise = torch.normal(0, noise_std, size=gradients.shape, device=gradients.device)
        return gradients + noise
    
    def privatize_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """Apply DP: clip + noise."""
        clipped = self.clip_gradients(gradients)
        return self.add_noise(clipped)


class RenyiDPAccountant:
    """Rényi DP accountant for privacy budget tracking."""
    
    def __init__(self, epsilon: float, delta: float, sampling_rate: float):
        self.target_epsilon = epsilon
        self.delta = delta
        self.sampling_rate = sampling_rate
        self.steps = 0
        self.alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    
    def _compute_rdp(self, sigma: float, alpha: float) -> float:
        """Compute Rényi DP for single step."""
        return alpha / (2 * sigma ** 2)
    
    def get_privacy_spent(self, sigma: float, steps: int) -> Tuple[float, float]:
        """Compute privacy spent after given steps."""
        rdp_values = []
        for alpha in self.alphas:
            rdp_step = self._compute_rdp(sigma, alpha)
            rdp_subsampled = self.sampling_rate * rdp_step
            rdp_total = steps * rdp_subsampled
            rdp_values.append(rdp_total)
        
        epsilon_values = [
            rdp + math.log(1 / self.delta) / (alpha - 1)
            for rdp, alpha in zip(rdp_values, self.alphas)
        ]
        
        return min(epsilon_values), self.delta
    
    def step(self):
        """Record a training step."""
        self.steps += 1
    
    def get_current_epsilon(self, sigma: float) -> float:
        """Get current epsilon value."""
        epsilon, _ = self.get_privacy_spent(sigma, self.steps)
        return epsilon