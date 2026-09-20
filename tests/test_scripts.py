"""Tests for helper functions in the experiment and data-preparation scripts."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestExperimentHelpers:
    def test_required_depths_bracket_restart_cost(self):
        exp = _load("experiment_adaptive_depth")
        depths = exp.required_fixed_depths([1, 2, 4, 8], d_lo=4, d_hi=8, fractions=[0.75])
        assert max(depths) >= 4 + 0.75 * 8
        assert 1 in depths and 8 in depths

    def test_required_depths_use_deep_cost(self):
        exp = _load("experiment_adaptive_depth")
        # d_hi=8 but escalated tokens cost 9 (pause work) -> restart at f=0.5 is 2+4.5=6.5, fine;
        # at f=1.0-ish fractions the ceiling of d_lo + f*deep_cost must be added.
        depths = exp.required_fixed_depths([1, 2, 4, 8], 2, 8, [0.9], deep_cost=9.0)
        assert max(depths) >= 2 + 0.9 * 9.0

    def test_required_depths_unchanged_when_covered(self):
        exp = _load("experiment_adaptive_depth")
        assert exp.required_fixed_depths([1, 2, 4, 8], 2, 8, [0.1, 0.5]) == [1, 2, 4, 8]

    def test_interp_refuses_to_extrapolate(self):
        exp = _load("experiment_adaptive_depth")
        curve = {1: 5.0, 2: 4.0, 4: 3.5}
        assert exp.interp_fixed(curve, 2) == pytest.approx(4.0)
        assert 3.5 < exp.interp_fixed(curve, 3) < 4.0
        with pytest.raises(ValueError):
            exp.interp_fixed(curve, 6)


class TestPrepareDataHelpers:
    def test_choose_val_docs(self):
        pytest.importorskip("tokenizers")
        prep = _load("prepare_data")
        assert prep.choose_val_docs(10_000, 50_000) == 10_000
        assert prep.choose_val_docs(10_000, 20) == 2
        assert prep.choose_val_docs(5, 5) == 1
        assert prep.choose_val_docs(1, 2) == 1
        with pytest.raises(ValueError):
            prep.choose_val_docs(1, 1)
