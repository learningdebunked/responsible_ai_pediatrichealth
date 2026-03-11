"""Tests for privacy modules (DP mechanisms and secure aggregation)."""

import pytest
import torch
import numpy as np
from src.privacy.dp_mechanisms import DPMechanism, RenyiDPAccountant, GradientInversionDefense
from src.privacy.secure_aggregation import (
    SecureAggregator, FederatedAveraging, FederatedProximal
)


class TestDPMechanism:
    """Tests for differential privacy mechanism."""

    def test_clip_gradients_norm(self):
        dp = DPMechanism(epsilon=1.0, delta=1e-5, clip_norm=1.0)
        grad = torch.randn(100) * 10  # Large gradients
        clipped = dp.clip_gradients(grad)
        assert torch.norm(clipped, p=2) <= 1.0 + 1e-5

    def test_clip_gradients_no_clip_when_small(self):
        dp = DPMechanism(epsilon=1.0, delta=1e-5, clip_norm=100.0)
        grad = torch.ones(10) * 0.1
        clipped = dp.clip_gradients(grad)
        assert torch.allclose(clipped, grad, atol=1e-6)

    def test_add_noise_changes_gradients(self):
        dp = DPMechanism(epsilon=1.0, delta=1e-5, clip_norm=1.0)
        grad = torch.zeros(100)
        noisy = dp.add_noise(grad)
        assert not torch.allclose(noisy, grad)

    def test_privatize_gradients_pipeline(self):
        dp = DPMechanism(epsilon=1.0, delta=1e-5, clip_norm=1.0)
        grad = torch.randn(50) * 5
        private = dp.privatize_gradients(grad)
        assert private.shape == grad.shape

    def test_noise_multiplier_positive(self):
        dp = DPMechanism(epsilon=0.5, delta=1e-5, clip_norm=1.0)
        assert dp.noise_multiplier > 0


class TestRenyiDPAccountant:
    """Tests for Renyi DP accountant."""

    def test_privacy_spent_increases_with_steps(self):
        acc = RenyiDPAccountant(epsilon=1.0, delta=1e-5, sampling_rate=0.01)
        eps1, _ = acc.get_privacy_spent(sigma=1.0, steps=10)
        eps2, _ = acc.get_privacy_spent(sigma=1.0, steps=100)
        assert eps2 > eps1

    def test_step_increments(self):
        acc = RenyiDPAccountant(epsilon=1.0, delta=1e-5, sampling_rate=0.01)
        assert acc.steps == 0
        acc.step()
        acc.step()
        assert acc.steps == 2

    def test_current_epsilon(self):
        acc = RenyiDPAccountant(epsilon=1.0, delta=1e-5, sampling_rate=0.01)
        acc.step()
        eps = acc.get_current_epsilon(sigma=1.0)
        assert eps > 0


class TestGradientInversionDefense:
    """Tests for gradient inversion defense."""

    @pytest.fixture
    def defense(self):
        return GradientInversionDefense(
            clip_norm=1.0, compression_ratio=0.1, perturbation_strength=0.1
        )

    def test_gradient_compression_sparsifies(self, defense):
        grad = torch.randn(100)
        compressed = defense.gradient_compression(grad)
        # Only ~10% should be non-zero
        n_nonzero = (compressed.abs() > 1e-7).sum().item()
        assert n_nonzero <= 15  # ~10% with some tolerance

    def test_representation_perturbation_2d(self, defense):
        grad = torch.randn(10, 10)
        perturbed = defense.representation_perturbation(grad)
        assert perturbed.shape == grad.shape
        assert not torch.allclose(perturbed, grad)

    def test_representation_perturbation_1d(self, defense):
        grad = torch.randn(50)
        perturbed = defense.representation_perturbation(grad)
        assert perturbed.shape == grad.shape

    def test_instahide_mix(self, defense):
        grads = [torch.randn(20) for _ in range(5)]
        mixed = defense.instahide_mix(grads, n_mix=2)
        assert len(mixed) == 5
        # Mixed gradients should differ from originals
        assert not torch.allclose(mixed[0], grads[0])

    def test_apply_all_defenses(self, defense):
        grad = torch.randn(10, 10) * 5
        defended = defense.apply_all_defenses(grad)
        assert defended.shape == grad.shape

    def test_estimate_reconstruction_risk(self, defense):
        grad = torch.randn(10, 10)
        risk = defense.estimate_reconstruction_risk(grad, batch_size=32)
        assert 'risk_level' in risk
        assert risk['risk_level'] in ['low', 'medium', 'high']
        assert 'sparsity' in risk
        assert 'effective_rank' in risk


class TestSecureAggregator:
    """Tests for secure aggregation."""

    @pytest.fixture
    def aggregator(self):
        return SecureAggregator(num_clients=3, seed=42)

    def test_generate_mask_deterministic(self, aggregator):
        m1 = aggregator.generate_mask("client_0", 1, (10,))
        m2 = aggregator.generate_mask("client_0", 1, (10,))
        assert torch.allclose(m1, m2)

    def test_generate_mask_different_clients(self, aggregator):
        m1 = aggregator.generate_mask("client_0", 1, (10,))
        m2 = aggregator.generate_mask("client_1", 1, (10,))
        assert not torch.allclose(m1, m2)

    def test_mask_model_update(self, aggregator):
        update = {'weight': torch.ones(5), 'bias': torch.zeros(3)}
        masked = aggregator.mask_model_update(update, "client_0", 1)
        assert set(masked.keys()) == set(update.keys())
        # Masked should differ from original
        assert not torch.allclose(masked['weight'], update['weight'])

    def test_aggregate_masked_updates_cancels_masks(self, aggregator):
        updates = [
            {'w': torch.ones(5) * (i + 1)} for i in range(3)
        ]
        client_ids = [f"client_{i}" for i in range(3)]

        # Mask each update
        masked = [
            aggregator.mask_model_update(u, cid, 1)
            for u, cid in zip(updates, client_ids)
        ]

        # Aggregate
        result = aggregator.aggregate_masked_updates(masked, client_ids, 1)

        # Expected: mean of [1, 2, 3] = 2.0
        expected = torch.ones(5) * 2.0
        assert torch.allclose(result['w'], expected, atol=1e-5)

    def test_verify_aggregation_integrity(self, aggregator):
        updates = [{'w': torch.randn(5)} for _ in range(3)]
        aggregated = aggregator.simple_aggregate(updates)
        results = aggregator.verify_aggregation_integrity(
            updates, aggregated
        )
        for name, info in results.items():
            assert info['passed'], f"Integrity check failed for {name}"

    def test_dropped_client_warning(self, aggregator):
        updates = [{'w': torch.ones(5)} for _ in range(2)]
        client_ids = ["client_0", "client_1"]
        expected_ids = ["client_0", "client_1", "client_2"]

        with pytest.warns(UserWarning, match="Dropped clients"):
            aggregator.aggregate_masked_updates(
                updates, client_ids, 1,
                expected_client_ids=expected_ids
            )


class TestFederatedAveraging:
    """Tests for FedAvg."""

    def test_aggregate_weighted_average(self):
        models = [
            {'w': torch.ones(3) * 1.0},
            {'w': torch.ones(3) * 3.0},
        ]
        sizes = [100, 300]
        result = FederatedAveraging.aggregate(models, sizes)
        # Expected: (100*1 + 300*3) / 400 = 2.5
        assert torch.allclose(result['w'], torch.ones(3) * 2.5)


class TestFederatedProximal:
    """Tests for FedProx."""

    def test_aggregate_with_proximal_term(self):
        fedprox = FederatedProximal(mu=0.5)
        client_models = [{'w': torch.ones(3) * 2.0}]
        global_model = {'w': torch.ones(3) * 1.0}
        result = fedprox.aggregate(client_models, [100], global_model)
        # Expected: (1-0.5)*2.0 + 0.5*1.0 = 1.5
        assert torch.allclose(result['w'], torch.ones(3) * 1.5)
