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


class GradientInversionDefense:
    """Analyze and defend against gradient inversion attacks.

    Gradient inversion attacks (Zhu et al., 2019; Geiping et al., 2020)
    attempt to reconstruct training data from shared gradients in
    federated learning. This class quantifies vulnerability and applies
    defense mechanisms beyond standard DP noise.

    Defense strategies:
    1. Gradient compression (top-k sparsification)
    2. Gradient perturbation (Soteria-style representation perturbation)
    3. InstaHide-style mixing of gradient signals
    """

    def __init__(self, clip_norm: float = 1.0, compression_ratio: float = 0.1,
                 perturbation_strength: float = 0.1):
        """Initialize gradient inversion defense.

        Args:
            clip_norm: Maximum gradient norm
            compression_ratio: Fraction of gradient values to keep (top-k)
            perturbation_strength: Strength of representation perturbation
        """
        self.clip_norm = clip_norm
        self.compression_ratio = compression_ratio
        self.perturbation_strength = perturbation_strength

    def gradient_compression(self, gradients: torch.Tensor) -> torch.Tensor:
        """Apply top-k gradient sparsification.

        Keeps only the top compression_ratio fraction of gradient values
        by magnitude, zeroing out the rest. This makes gradient inversion
        significantly harder while preserving learning signal.

        Args:
            gradients: Raw gradient tensor

        Returns:
            Sparsified gradient tensor
        """
        flat = gradients.flatten()
        k = max(1, int(len(flat) * self.compression_ratio))

        # Find top-k threshold
        values, _ = torch.topk(flat.abs(), k)
        threshold = values[-1]

        # Zero out values below threshold
        mask = flat.abs() >= threshold
        compressed = flat * mask.float()

        return compressed.reshape(gradients.shape)

    def representation_perturbation(self, gradients: torch.Tensor,
                                     layer_index: int = -1) -> torch.Tensor:
        """Apply Soteria-style representation perturbation.

        Perturbs gradients in directions that maximally damage
        reconstruction quality while minimally affecting learning.

        Args:
            gradients: Gradient tensor
            layer_index: Which layer's gradients these are (-1 = last)

        Returns:
            Perturbed gradient tensor
        """
        # SVD-based perturbation: project out top singular vectors
        # that carry the most reconstructable information
        if gradients.dim() < 2:
            # For 1D gradients, add simple noise
            noise = torch.randn_like(gradients) * self.perturbation_strength
            return gradients + noise

        try:
            U, S, Vh = torch.linalg.svd(gradients, full_matrices=False)

            # Zero out top singular values (most reconstructable directions)
            n_perturb = max(1, int(len(S) * self.perturbation_strength))
            S_perturbed = S.clone()
            S_perturbed[:n_perturb] = 0

            perturbed = U @ torch.diag(S_perturbed) @ Vh
            return perturbed
        except RuntimeError:
            # SVD may fail on some tensor shapes; fall back to noise
            noise = torch.randn_like(gradients) * self.perturbation_strength
            return gradients + noise

    def instahide_mix(self, gradients_list: list,
                       n_mix: int = 2) -> list:
        """Mix gradient signals across clients (InstaHide-style).

        Combines gradients from multiple clients with random signs,
        making it impossible to attribute specific gradient components
        to individual clients.

        Args:
            gradients_list: List of gradient tensors from different clients
            n_mix: Number of clients to mix per output

        Returns:
            List of mixed gradient tensors (same length as input)
        """
        n_clients = len(gradients_list)
        if n_clients < n_mix:
            return gradients_list

        mixed = []
        for i in range(n_clients):
            # Select n_mix random clients (including self)
            indices = [i]
            others = torch.randperm(n_clients)[:n_mix - 1].tolist()
            indices.extend([j for j in others if j != i][:n_mix - 1])

            # Random mixing coefficients (sum to 1)
            coeffs = torch.rand(len(indices))
            coeffs = coeffs / coeffs.sum()

            # Random signs
            signs = torch.sign(torch.randn(len(indices)))

            # Mix
            result = sum(
                c * s * gradients_list[j]
                for c, s, j in zip(coeffs, signs, indices)
            )
            mixed.append(result)

        return mixed

    def apply_all_defenses(self, gradients: torch.Tensor) -> torch.Tensor:
        """Apply all defense mechanisms in sequence.

        Args:
            gradients: Raw gradient tensor

        Returns:
            Defended gradient tensor
        """
        # 1. Clip
        grad_norm = torch.norm(gradients, p=2)
        clip_factor = min(1.0, self.clip_norm / (grad_norm + 1e-6))
        defended = gradients * clip_factor

        # 2. Compress
        defended = self.gradient_compression(defended)

        # 3. Perturb
        defended = self.representation_perturbation(defended)

        return defended

    def estimate_reconstruction_risk(self, gradients: torch.Tensor,
                                      batch_size: int = 1) -> dict:
        """Estimate how vulnerable gradients are to inversion attacks.

        Computes metrics that correlate with reconstruction quality:
        - Gradient sparsity (sparser = harder to reconstruct)
        - Effective rank (lower = easier to reconstruct)
        - SNR after defenses (lower = better defense)

        Args:
            gradients: Gradient tensor to analyze
            batch_size: Training batch size (larger = harder to reconstruct)

        Returns:
            Dict with risk metrics and qualitative assessment
        """
        flat = gradients.flatten().float()

        # Sparsity
        sparsity = float((flat.abs() < 1e-7).float().mean())

        # Effective rank (via SVD if 2D)
        if gradients.dim() >= 2:
            try:
                S = torch.linalg.svdvals(gradients.float())
                S_norm = S / S.sum()
                entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-10))
                effective_rank = float(torch.exp(entropy))
            except RuntimeError:
                effective_rank = float(min(gradients.shape))
        else:
            effective_rank = float(len(flat))

        # SNR estimate
        signal_power = float(flat.var())
        defended = self.apply_all_defenses(gradients)
        noise_power = float((defended.flatten() - flat).var())
        snr = signal_power / (noise_power + 1e-10)

        # Risk assessment
        if batch_size >= 32 and sparsity > 0.5:
            risk_level = 'low'
        elif batch_size >= 8 or sparsity > 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'high'

        return {
            'sparsity': sparsity,
            'effective_rank': effective_rank,
            'snr_after_defense': float(snr),
            'batch_size': batch_size,
            'risk_level': risk_level,
        }