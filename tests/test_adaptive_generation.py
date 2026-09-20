"""Tests for the uncertainty-triggered generation loop and threshold calibration."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from lantern.controller.adaptive_generation import (
    AdaptiveGenerationConfig,
    AdaptiveGenerator,
    calibrate_threshold,
    collect_signals,
)
from lantern.models.lantern_model import LANTERNModel
from lantern.utils.config import create_small_config


def _model(vocab=80):
    torch.manual_seed(0)
    cfg = create_small_config()
    cfg.vocab_size = vocab
    cfg.max_position = 64
    cfg.dropout = 0.0
    return LANTERNModel(cfg).eval(), cfg


class _DictLoader:
    def __init__(self, x):
        self.ds = TensorDataset(x[:, :-1], x[:, 1:])

    def __iter__(self):
        for a, b in DataLoader(self.ds, batch_size=2):
            yield {"input_ids": a, "labels": b}


class TestAdaptiveGenerator:
    @pytest.mark.parametrize("signal", ["entropy", "probe", "step_kl"])
    def test_escalates_everything_with_low_threshold(self, signal):
        m, cfg = _model()
        x = torch.randint(0, cfg.vocab_size, (2, 5))
        gen = AdaptiveGenerator(m, AdaptiveGenerationConfig(
            max_new_tokens=6, temperature=0, signal=signal, threshold=-1.0, pause_steps=1,
        ))
        r = gen.generate(x)
        assert r.tokens.shape == (2, 11)
        assert r.escalation_rate == 1.0
        assert all(t.depth == cfg.steps_reasoning and t.pause_steps == 1 for row in r.trace for t in row)

    def test_never_escalates_with_inf_threshold_and_matches_fixed_depth(self):
        m, cfg = _model()
        x = torch.randint(0, cfg.vocab_size, (2, 5))
        r = AdaptiveGenerator(m, AdaptiveGenerationConfig(
            max_new_tokens=8, temperature=0, signal="entropy",
        )).generate(x)
        fixed = m.generate(x, max_new_tokens=8, temperature=0)
        assert r.escalation_rate == 0.0
        assert torch.equal(r.tokens, fixed)

    def test_always_escalate_matches_fixed_deep_generation(self):
        """With every token escalated, output equals fixed-depth generation at steps_deep."""
        m, cfg = _model()
        x = torch.randint(0, cfg.vocab_size, (1, 5))
        r = AdaptiveGenerator(m, AdaptiveGenerationConfig(
            max_new_tokens=8, temperature=0, signal="entropy", threshold=-1.0, pause_steps=2,
        )).generate(x)
        fixed = m.generate(x, max_new_tokens=8, temperature=0,
                           steps_per_block=cfg.steps_reasoning, pause_steps=2)
        assert torch.equal(r.tokens, fixed)

    def test_partial_escalation_in_batch_keeps_other_rows_intact(self):
        """A row that never escalates must produce the same tokens as when generated alone."""
        m, cfg = _model()
        torch.manual_seed(1)
        x = torch.randint(0, cfg.vocab_size, (2, 5))
        cfg_gen = AdaptiveGenerationConfig(max_new_tokens=6, temperature=0, signal="entropy")
        # Find a threshold that escalates row 0 but not row 1 on the first step.
        solo = AdaptiveGenerator(m, cfg_gen).generate(x[1:2])
        r_solo_sig = [t.entropy for t in solo.trace[0]]
        thr = max(r_solo_sig) + 1e-3
        row0 = AdaptiveGenerator(m, AdaptiveGenerationConfig(
            max_new_tokens=6, temperature=0, signal="entropy", threshold=-1.0)).generate(x[0:1])
        both = AdaptiveGenerator(m, AdaptiveGenerationConfig(
            max_new_tokens=6, temperature=0, signal="entropy", threshold=thr)).generate(x)
        assert all(not t.escalated for t in both.trace[1])
        assert torch.equal(both.tokens[1], solo.tokens[0])
        if all(t.escalated for t in both.trace[0]):
            assert torch.equal(both.tokens[0], row0.tokens[0])

    def test_deep_pass_runs_only_on_escalated_rows(self, monkeypatch):
        """The deep forward must see only the escalated rows of the batch."""
        m, cfg = _model()
        torch.manual_seed(2)
        x = torch.randint(0, cfg.vocab_size, (3, 5))
        seen = []
        orig_forward = m.forward

        def spy(input_ids, *a, **kw):
            if kw.get("steps_per_block") == cfg.steps_reasoning:
                seen.append(input_ids.shape[0])
            return orig_forward(input_ids, *a, **kw)

        monkeypatch.setattr(m, "forward", spy)
        # Escalate only the highest-entropy row on the first step.
        probe = AdaptiveGenerator(m, AdaptiveGenerationConfig(max_new_tokens=1, temperature=0)).generate(x)
        ents = sorted((row[0].entropy for row in probe.trace), reverse=True)
        thr = (ents[0] + ents[1]) / 2
        seen.clear()
        r = AdaptiveGenerator(m, AdaptiveGenerationConfig(
            max_new_tokens=1, temperature=0, signal="entropy", threshold=thr,
        )).generate(x)
        assert sum(t[0].escalated for t in r.trace) == 1
        assert seen == [1]

    def test_eos_handling(self):
        m, cfg = _model()
        m.lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=True)
        with torch.no_grad():
            m.lm_head.bias.zero_()
            m.lm_head.bias[3] = 100.0
        x = torch.randint(0, cfg.vocab_size, (2, 4))
        r = AdaptiveGenerator(m, AdaptiveGenerationConfig(
            max_new_tokens=10, temperature=0, eos_token_id=3, signal="none",
        )).generate(x)
        assert r.tokens.shape[1] == 5
        assert len(r.trace[0]) == 1

    def test_trace_records_clamped_pause_steps(self):
        m, cfg = _model()
        x = torch.randint(0, cfg.vocab_size, (1, 4))
        r = AdaptiveGenerator(m, AdaptiveGenerationConfig(
            max_new_tokens=3, temperature=0, signal="entropy", threshold=-1.0,
            pause_steps=cfg.max_pause_steps + 5,
        )).generate(x)
        assert all(t.pause_steps == cfg.max_pause_steps for t in r.trace[0])

    def test_negative_pause_steps_rejected(self):
        m, _ = _model()
        with pytest.raises(ValueError):
            AdaptiveGenerator(m, AdaptiveGenerationConfig(pause_steps=-1))

    def test_bad_signal_rejected(self):
        m, _ = _model()
        with pytest.raises(ValueError):
            AdaptiveGenerator(m, AdaptiveGenerationConfig(signal="vibes"))


class TestCalibration:
    def test_collect_and_calibrate(self):
        m, cfg = _model()
        x = torch.randint(0, cfg.vocab_size, (4, 9))
        sig = collect_signals(m, _DictLoader(x))
        assert set(sig) == {"entropy", "probe", "step_kl"}
        assert all(v.shape == (32,) for v in sig.values())
        thr = calibrate_threshold(sig["entropy"], 0.25)
        frac = (sig["entropy"] > thr).float().mean().item()
        assert 0.15 <= frac <= 0.35

    def test_constant_signal_rejected(self):
        with pytest.raises(ValueError):
            calibrate_threshold(torch.zeros(10), 0.2)

    def test_step_kl_needs_two_steps(self):
        m, _ = _model()
        x = torch.randint(0, 80, (1, 4))
        gen = AdaptiveGenerator(m, AdaptiveGenerationConfig(signal="step_kl", steps_base=1, threshold=0.0))
        with pytest.raises(ValueError):
            gen.generate(x)

    def test_bad_fraction(self):
        with pytest.raises(ValueError):
            calibrate_threshold(torch.arange(10.0), 1.0)
