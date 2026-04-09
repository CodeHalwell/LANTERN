"""Tests for adaptive EMA thresholds in UncertaintyController."""

import pytest
import torch

from lantern.controller.uncertainty_controller import (
    UncertaintyController,
    UncertaintyLevel,
)


class TestAdaptiveEMAThresholds:
    """Tests for EMA update, adaptive thresholds, safe variance, and adaptation."""

    def test_ema_initial_state(self):
        """EMA should start uninitialized."""
        ctrl = UncertaintyController()
        assert ctrl._ema_initialized is False

    def test_ema_updates_after_update_call(self):
        """Calling _update_ema should initialise and update the running stats."""
        ctrl = UncertaintyController(ema_decay=0.9)
        batch = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        ctrl._update_ema(batch)
        assert ctrl._ema_initialized is True
        assert ctrl._ema_mean is not None
        assert ctrl._ema_var is not None
        # After first batch, EMA mean should approximate batch mean
        assert abs(ctrl._ema_mean - batch.mean().item()) < 1.0

    def test_ema_mean_tracks_distribution(self):
        """After many updates with similar data, EMA mean should converge."""
        ctrl = UncertaintyController(ema_decay=0.9)
        for _ in range(100):
            batch = torch.randn(32) * 0.5 + 2.0  # mean ≈ 2.0
            ctrl._update_ema(batch)
        assert ctrl._ema_mean is not None
        assert abs(ctrl._ema_mean - 2.0) < 0.5

    def test_get_adaptive_thresholds_before_ema(self):
        """Before EMA is initialized, should return static thresholds."""
        ctrl = UncertaintyController(tau_low=1.0, tau_mid=2.0, tau_high=3.0)
        low, mid, high = ctrl._get_adaptive_thresholds()
        assert low == 1.0
        assert mid == 2.0
        assert high == 3.0

    def test_get_adaptive_thresholds_after_ema(self):
        """After EMA initialization, thresholds should be Z-score based."""
        ctrl = UncertaintyController(ema_decay=0.9)
        batch = torch.randn(64) * 1.0 + 5.0  # mean ≈ 5.0, std ≈ 1.0
        ctrl._update_ema(batch)
        low, mid, high = ctrl._get_adaptive_thresholds()
        # With mean ≈ 5.0 and std ≈ 1.0:
        # tau_low ≈ 5.0 - 0.52*1.0 ≈ 4.48
        # tau_mid ≈ 5.0 + 0.52*1.0 ≈ 5.52
        # tau_high ≈ 5.0 + 1.28*1.0 ≈ 6.28
        assert low < mid < high
        # Rough range checks
        assert abs(low - 4.48) < 1.0
        assert abs(mid - 5.52) < 1.0
        assert abs(high - 6.28) < 1.0

    def test_safe_variance_single_element(self):
        """_update_ema with a single-element tensor should not produce NaN."""
        ctrl = UncertaintyController(ema_decay=0.9)
        single = torch.tensor([3.14])
        ctrl._update_ema(single)
        assert ctrl._ema_initialized is True
        assert ctrl._ema_var is not None
        # Variance for a single element should be 0 (safe variance)
        assert ctrl._ema_var == 0.0 or not torch.tensor(ctrl._ema_var).isnan()

    def test_safe_variance_repeated_updates(self):
        """Repeated single-element updates should not accumulate NaN."""
        ctrl = UncertaintyController(ema_decay=0.9)
        for i in range(20):
            ctrl._update_ema(torch.tensor([float(i)]))
        assert not torch.tensor(ctrl._ema_mean).isnan()
        assert not torch.tensor(ctrl._ema_var).isnan()

    def test_thresholds_adapt_over_many_updates(self):
        """Thresholds should change as EMA statistics accumulate."""
        ctrl = UncertaintyController(ema_decay=0.95)
        # First phase: low uncertainty
        for _ in range(50):
            ctrl._update_ema(torch.randn(16) * 0.1 + 0.5)
        low1, mid1, high1 = ctrl._get_adaptive_thresholds()

        # Second phase: high uncertainty
        for _ in range(200):
            ctrl._update_ema(torch.randn(16) * 0.1 + 5.0)
        low2, mid2, high2 = ctrl._get_adaptive_thresholds()

        # Thresholds should have shifted upward
        assert low2 > low1
        assert mid2 > mid1
        assert high2 > high1

    def test_classify_uses_adaptive_thresholds(self):
        """classify_uncertainty should use adaptive thresholds after EMA warm-up."""
        ctrl = UncertaintyController(ema_decay=0.9)
        # Warm up with values around 5.0
        for _ in range(100):
            ctrl._update_ema(torch.randn(32) * 0.5 + 5.0)

        # A score far below the mean should be CONFIDENT
        low_score = torch.tensor(2.0)
        level = ctrl.classify_uncertainty(low_score)
        assert level == UncertaintyLevel.CONFIDENT

        # A score far above the mean should be HIGH or VERY_HIGH
        high_score = torch.tensor(8.0)
        level = ctrl.classify_uncertainty(high_score)
        assert level in (UncertaintyLevel.HIGH, UncertaintyLevel.VERY_HIGH)

    def test_threshold_ordering_always_holds(self):
        """tau_low < tau_mid < tau_high should always hold."""
        ctrl = UncertaintyController(ema_decay=0.9)
        # Random updates
        for _ in range(50):
            ctrl._update_ema(torch.randn(8) * 2.0 + 1.0)
        low, mid, high = ctrl._get_adaptive_thresholds()
        assert low < mid < high


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
