"""
Tests for LANTERN three-phase training curriculum.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from lantern.models.lantern_model import LANTERNModel
from lantern.utils.config import create_small_config
from lantern.training import (
    Phase1Trainer,
    Phase2Trainer,
    Phase3Trainer,
    selective_dropout_train,
    compute_ponder_cost_masked,
)


TEST_VOCAB_SIZE = 100


def _make_dataloader(vocab_size=TEST_VOCAB_SIZE, batch_size=2, seq_len=16, n=8):
    """Create a small synthetic dataloader for testing."""
    input_ids = torch.randint(0, vocab_size, (n, seq_len))
    labels = torch.randint(0, vocab_size, (n, seq_len))
    dataset = TensorDataset(input_ids, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: {
            "input_ids": torch.stack([b[0] for b in batch]),
            "labels": torch.stack([b[1] for b in batch]),
        },
    )


class TestSelectiveDropoutTrain:
    """Tests for selective_dropout_train context manager."""

    def test_only_dropout_in_train_mode(self):
        """Only nn.Dropout layers should be in train mode."""
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Dropout(0.1),
            nn.LayerNorm(10),
            nn.Linear(10, 10),
        )
        model.eval()

        with selective_dropout_train(model):
            assert model[1].training  # Dropout should be in train
            assert not model[0].training  # Linear should be eval
            assert not model[2].training  # LayerNorm should be eval

        # Should restore
        assert not model[1].training

    def test_restores_original_state(self):
        """State should be fully restored after context manager."""
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Dropout(0.1),
        )
        model.train()

        with selective_dropout_train(model):
            pass

        assert model.training


class TestComputePonderCostMasked:
    """Tests for ponder cost masking."""

    def test_no_mask_uses_mean(self):
        """Without mask, should return mean."""
        cost = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = compute_ponder_cost_masked(cost, attention_mask=None)
        assert torch.isclose(result, cost.mean())

    def test_mask_excludes_padding(self):
        """Padding tokens should be excluded from mean."""
        cost = torch.tensor([[1.0, 2.0, 100.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])
        result = compute_ponder_cost_masked(cost, mask)
        expected = (1.0 + 2.0) / 2.0  # Only real tokens
        assert torch.isclose(result, torch.tensor(expected))

    def test_all_padding_no_division_by_zero(self):
        """All-padding batch should not cause division by zero."""
        cost = torch.tensor([[1.0, 2.0]])
        mask = torch.tensor([[0.0, 0.0]])
        result = compute_ponder_cost_masked(cost, mask)
        assert not torch.isnan(result)


class TestPhase1Trainer:
    """Tests for Phase 1 backbone pretraining."""

    def test_train_step_runs(self):
        """Phase 1 train step should run without error."""
        config = create_small_config()
        config.vocab_size = TEST_VOCAB_SIZE
        model = LANTERNModel(config)
        loader = _make_dataloader(vocab_size=config.vocab_size)

        trainer = Phase1Trainer(model, loader, max_steps=2)
        batch = next(iter(loader))
        loss = trainer.train_step(batch)
        assert isinstance(loss, float)
        assert loss > 0

    def test_random_depth_variation(self):
        """Phase 1 should use random depths (no crash with any depth)."""
        config = create_small_config()
        config.vocab_size = TEST_VOCAB_SIZE
        model = LANTERNModel(config)
        loader = _make_dataloader(vocab_size=config.vocab_size)

        trainer = Phase1Trainer(model, loader, max_steps=2)
        batch = next(iter(loader))

        # Run multiple steps - random depth should not crash
        for _ in range(5):
            loss = trainer.train_step(batch)
            assert loss > 0


class TestPhase2Trainer:
    """Tests for Phase 2 probe distillation."""

    def test_backbone_frozen(self):
        """Backbone should be frozen, probe should be trainable."""
        config = create_small_config()
        config.vocab_size = TEST_VOCAB_SIZE
        model = LANTERNModel(config)
        loader = _make_dataloader(vocab_size=config.vocab_size)

        trainer = Phase2Trainer(model, loader, max_steps=1)

        # Backbone params should be frozen
        for name, param in model.named_parameters():
            if "epistemic_probe" in name:
                assert param.requires_grad
            else:
                assert not param.requires_grad

    def test_train_step_runs(self):
        """Phase 2 train step should run without error."""
        config = create_small_config()
        config.vocab_size = TEST_VOCAB_SIZE
        model = LANTERNModel(config)
        loader = _make_dataloader(vocab_size=config.vocab_size)

        trainer = Phase2Trainer(model, loader, num_mc_samples=2, max_steps=1)
        batch = next(iter(loader))
        loss = trainer.train_step(batch)
        assert isinstance(loss, float)

    def test_cleanup_unfreezes(self):
        """Cleanup should unfreeze all parameters."""
        config = create_small_config()
        config.vocab_size = TEST_VOCAB_SIZE
        model = LANTERNModel(config)
        loader = _make_dataloader(vocab_size=config.vocab_size)

        trainer = Phase2Trainer(model, loader, max_steps=1)
        trainer.cleanup()

        for param in model.parameters():
            assert param.requires_grad


class TestPhase3Trainer:
    """Tests for Phase 3 controller unlock."""

    def test_differential_lr(self):
        """Optimizer should have two param groups with different LRs."""
        config = create_small_config()
        config.vocab_size = TEST_VOCAB_SIZE
        config.use_adaptive_halting = True
        model = LANTERNModel(config)
        loader = _make_dataloader(vocab_size=config.vocab_size)

        trainer = Phase3Trainer(
            model, loader,
            backbone_lr=1e-5, reasoning_lr=1e-3,
            use_bfloat16=False,
        )

        assert len(trainer.optimizer.param_groups) == 2
        assert trainer.optimizer.param_groups[0]["lr"] == 1e-5
        assert trainer.optimizer.param_groups[1]["lr"] == 1e-3

    def test_ponder_warmup(self):
        """λ should start at 0 and ramp linearly."""
        config = create_small_config()
        config.vocab_size = TEST_VOCAB_SIZE
        config.use_adaptive_halting = True
        model = LANTERNModel(config)
        loader = _make_dataloader(vocab_size=config.vocab_size)

        trainer = Phase3Trainer(
            model, loader,
            ponder_lambda=0.01,
            ponder_warmup_steps=100,
            use_bfloat16=False,
        )

        trainer.step = 0
        assert trainer._get_ponder_lambda() == 0.0

        trainer.step = 50
        assert abs(trainer._get_ponder_lambda() - 0.005) < 1e-6

        trainer.step = 100
        assert abs(trainer._get_ponder_lambda() - 0.01) < 1e-6

        trainer.step = 200
        assert abs(trainer._get_ponder_lambda() - 0.01) < 1e-6

    def test_train_step_runs(self):
        """Phase 3 train step should run without error."""
        config = create_small_config()
        config.vocab_size = TEST_VOCAB_SIZE
        config.use_adaptive_halting = True
        model = LANTERNModel(config)
        loader = _make_dataloader(vocab_size=config.vocab_size)

        trainer = Phase3Trainer(
            model, loader, max_steps=2,
            use_bfloat16=False,
        )
        batch = next(iter(loader))
        metrics = trainer.train_step(batch)

        assert "total_loss" in metrics
        assert "ce_loss" in metrics
        assert "ponder_cost" in metrics
        assert "ponder_lambda" in metrics
