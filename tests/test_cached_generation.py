"""Tests for the model-level cache, pause steps, step trace and generate()."""

import pytest
import torch

from lantern.models.lantern_model import LANTERNModel, sample_from_logits
from lantern.utils.config import create_300m_config, create_small_config


def _model(vocab=120, max_position=64, halting=False):
    torch.manual_seed(0)
    cfg = create_small_config()
    cfg.vocab_size = vocab
    cfg.max_position = max_position
    cfg.dropout = 0.0
    cfg.use_adaptive_halting = halting
    return LANTERNModel(cfg).eval(), cfg


class TestPauseSteps:
    def test_pause_changes_logits(self):
        m, cfg = _model()
        x = torch.randint(0, cfg.vocab_size, (2, 10))
        a, _, _ = m(x)
        b, _, _ = m(x, pause_steps=2)
        assert a.shape == b.shape
        assert not torch.allclose(a, b)

    def test_pause_gradient_reaches_pause_module(self):
        m, cfg = _model()
        m.train()
        x = torch.randint(0, cfg.vocab_size, (2, 10))
        logits, _, _ = m(x, pause_steps=1)
        logits.sum().backward()
        assert m.pause_module.ffn_w1.weight.grad is not None
        assert m.pause_module.pause_embeddings.weight.grad is not None


class TestStepTrace:
    def test_step_states_and_kl(self):
        m, cfg = _model()
        x = torch.randint(0, cfg.vocab_size, (2, 10))
        states = []
        logits, hidden, _ = m(x, steps_per_block=4, return_hidden_states=True, step_states=states)
        assert len(states) == 4
        kl = m.step_kl(states)
        assert kl.shape == (2,)
        assert (kl >= 0).all()
        assert m.step_kl(states, last_only=False).shape == (2, 10)
        assert torch.allclose(m.step_logits(states)[-1], logits[:, -1], atol=1e-5)
        assert m.probe_uncertainty(hidden).shape == (2, 10)

    def test_single_step_kl_is_zero(self):
        m, cfg = _model()
        x = torch.randint(0, cfg.vocab_size, (1, 5))
        states = []
        m(x, steps_per_block=1, step_states=states)
        assert torch.equal(m.step_kl(states), torch.zeros(1))


class TestCachedForward:
    def test_prefill_and_decode_match_full_forward(self):
        m, cfg = _model()
        x = torch.randint(0, cfg.vocab_size, (2, 12))
        full, _, _ = m(x, pause_steps=1)
        caches = m.create_kv_caches(2, 40, torch.device("cpu"))
        pre, _, _ = m(x[:, :8], pause_steps=1, kv_caches=caches, start_pos=0)
        assert torch.allclose(pre, full[:, :8], atol=1e-4)
        for t in range(8, 12):
            step, _, _ = m(x[:, t:t + 1], pause_steps=1, kv_caches=caches, start_pos=t)
            assert torch.allclose(step[:, 0], full[:, t], atol=1e-4)

    def test_variable_depth_uses_carry_forward(self):
        m, cfg = _model()
        x = torch.randint(0, cfg.vocab_size, (1, 9))
        caches = m.create_kv_caches(1, 16, torch.device("cpu"))
        m(x[:, :8], steps_per_block=1, kv_caches=caches, start_pos=0)
        c = caches[0]
        assert torch.equal(c.k_cache[3, :, :, :8], c.k_cache[0, :, :, :8])
        out, _, _ = m(x[:, 8:9], steps_per_block=4, kv_caches=caches, start_pos=8)
        assert torch.isfinite(out).all()

    def test_cached_forward_with_halting(self):
        m, cfg = _model(halting=True)
        x = torch.randint(0, cfg.vocab_size, (1, 6))
        caches = m.create_kv_caches(1, 16, torch.device("cpu"))
        out, _, ponder = m(x, use_adaptive_halting=True, kv_caches=caches)
        assert out.shape == (1, 6, cfg.vocab_size)
        assert ponder is not None


class TestGenerate:
    def test_cached_greedy_matches_uncached(self):
        m, cfg = _model()
        x = torch.randint(0, cfg.vocab_size, (2, 6))
        for pause in (0, 2):
            a = m.generate(x, max_new_tokens=15, temperature=0, use_cache=True, pause_steps=pause)
            b = m.generate(x, max_new_tokens=15, temperature=0, use_cache=False, pause_steps=pause)
            assert torch.equal(a, b)
            assert a.shape == (2, 21)

    def test_eos_stops_and_pads(self):
        m, cfg = _model()
        x = torch.randint(0, cfg.vocab_size, (2, 4))
        # Force EOS to be the argmax everywhere with a biased, untied LM head.
        m.lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=True)
        with torch.no_grad():
            m.lm_head.bias.zero_()
            m.lm_head.bias[7] = 100.0
        out = m.generate(x, max_new_tokens=10, temperature=0, eos_token_id=7)
        assert out.shape == (2, 5)
        assert (out[:, -1] == 7).all()

    def test_depth_beyond_max_steps_matches_full_forward(self):
        """generate() sizes its caches for the requested depth, so logits match a full forward."""
        m, cfg = _model()
        x = torch.randint(0, cfg.vocab_size, (2, 6))
        deep = cfg.max_steps + 3
        a = m.generate(x, max_new_tokens=8, temperature=0, steps_per_block=deep, use_cache=True)
        b = m.generate(x, max_new_tokens=8, temperature=0, steps_per_block=deep, use_cache=False)
        assert torch.equal(a, b)
        # And at the logit level, not just the argmax.
        full, _, _ = m(x, steps_per_block=deep)
        caches = m.create_kv_caches(2, 16, torch.device("cpu"), max_steps=deep)
        m(x[:, :4], steps_per_block=deep, kv_caches=caches, start_pos=0)
        step, _, _ = m(x[:, 4:5], steps_per_block=deep, kv_caches=caches, start_pos=4)
        assert torch.allclose(step[:, 0], full[:, 4], atol=1e-4)

    def test_undersized_cache_is_rejected(self):
        m, cfg = _model()
        x = torch.randint(0, cfg.vocab_size, (1, 4))
        caches = m.create_kv_caches(1, 16, torch.device("cpu"))  # config.max_steps slots
        with pytest.raises(ValueError):
            m(x, steps_per_block=cfg.max_steps + 1, kv_caches=caches)

    def test_respects_max_position(self):
        m, cfg = _model(max_position=16)
        x = torch.randint(0, cfg.vocab_size, (1, 12))
        out = m.generate(x, max_new_tokens=100, temperature=0)
        assert out.shape[1] == 16

    def test_sample_from_logits_filters(self):
        logits = torch.tensor([[0.0, 1.0, 2.0, 10.0]])
        assert sample_from_logits(logits, temperature=0).item() == 3
        torch.manual_seed(0)
        for _ in range(20):
            tok = sample_from_logits(logits, temperature=1.0, top_k=1).item()
            assert tok == 3


class TestCheckpointIO:
    def test_save_and_load_roundtrip_weights_only(self, tmp_path):
        from train import load_checkpoint, save_checkpoint

        m, cfg = _model()
        path = tmp_path / "ckpt.pt"
        save_checkpoint(m, path, phase=1, step=7, tokenizer_path="tok.json")
        loaded, ckpt = load_checkpoint(str(path), "cpu")
        assert ckpt["phase"] == 1 and ckpt["step"] == 7 and ckpt["tokenizer_path"] == "tok.json"
        assert loaded.config == cfg
        x = torch.randint(0, cfg.vocab_size, (1, 5))
        assert torch.allclose(m(x)[0], loaded.eval()(x)[0])

    def test_load_rejects_arbitrary_pickle(self, tmp_path):
        from train import load_checkpoint

        path = tmp_path / "bad.pt"
        torch.save({"config": {"x": object()}}, path)
        with pytest.raises(RuntimeError):
            load_checkpoint(str(path), "cpu")


class TestTrainHelpers:
    def test_make_loader_rejects_undersized_dataset(self):
        from torch.utils.data import TensorDataset

        from train import make_loader

        ds = TensorDataset(torch.zeros(3, 4))
        with pytest.raises(SystemExit):
            make_loader(ds, batch_size=8, shuffle=False, num_workers=0, device="cpu")
        assert make_loader(ds, batch_size=2, shuffle=False, num_workers=0, device="cpu") is not None

    def test_build_datasets_skips_undersized_val(self, tmp_path):
        import json
        import types

        import numpy as np

        from train import build_datasets

        np.arange(200, dtype=np.uint16).tofile(tmp_path / "train.bin")
        np.arange(5, dtype=np.uint16).tofile(tmp_path / "val.bin")
        (tmp_path / "meta.json").write_text(json.dumps({"vocab_size": 300, "eos_token_id": 2}))
        args = types.SimpleNamespace(data_dir=str(tmp_path), seq_length=16, data_path=None)
        train_ds, val_ds, vocab, _, eos = build_datasets(args, tmp_path)
        assert val_ds is None and len(train_ds) > 0 and vocab == 300 and eos == 2

    def test_evaluate_returns_nan_on_empty_loader(self):
        from train import evaluate

        m, _ = _model()
        val = evaluate(m, [], "cpu", max_batches=None)
        assert val != val  # NaN


class TestConfigs:
    def test_300m_config_size(self):
        cfg = create_300m_config()
        m = LANTERNModel(cfg)
        total = sum(p.numel() for p in m.parameters())
        assert 290e6 < total < 340e6
        assert 240e6 < m.get_num_params() < 290e6
        assert cfg.hidden_size % cfg.num_heads == 0
