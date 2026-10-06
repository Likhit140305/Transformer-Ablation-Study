"""Integrity checks for the ablation study. Run: python -m pytest research/tests"""

import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

from research.analysis.analyze_study import holm
from research.experiments.run_study import CONDITIONS
from research.experiments.train import lr_multiplier, resolved_model_config
from research.models.transformer.transformer import TransformerLM

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")

# The single model field each ablation is declared to change, and its value.
EXPECTED_CHANGE = {
    "abl_no_pos_enc": ("pos_encoding", "none"),
    "abl_sinusoidal_pos": ("pos_encoding", "sinusoidal"),
    "abl_single_head": ("n_heads", 1),
    "abl_reduced_ffn": ("d_ff", 256),
    "abl_linear_ffn": ("activation", "linear"),
    "abl_no_residual": ("use_residual", False),
    "abl_post_layernorm": ("pre_norm", False),
}


def resolved(experiment: str):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose("config", overrides=[f"experiment={experiment}"])
    return cfg, resolved_model_config(cfg)


def tiny_model(**overrides) -> TransformerLM:
    kwargs = dict(vocab_size=32, d_model=16, n_heads=4, n_layers=2, d_ff=32, max_seq_length=8,
                  dropout=0.0, emb_dropout=0.0, attn_dropout=0.0, ff_dropout=0.0)
    return TransformerLM(**{**kwargs, **overrides}).eval()


class ExperimentIntegrityTests(unittest.TestCase):
    def test_each_ablation_changes_exactly_its_declared_field(self):
        """Regression: overrides were once silently ignored and runs saved under 'baseline'."""
        _, base = resolved("baseline")
        self.assertEqual(set(EXPECTED_CHANGE) | {"baseline"}, set(CONDITIONS))
        for experiment, (field, value) in EXPECTED_CHANGE.items():
            cfg, model = resolved(experiment)
            self.assertEqual(cfg.experiment_name, experiment)
            changed = {k for k in base if base[k] != model[k]}
            self.assertEqual(changed, {field}, experiment)
            self.assertEqual(model[field], value, experiment)

    def test_every_ablation_builds_and_is_causal(self):
        torch.manual_seed(0)
        first, second = torch.tensor([[1, 2, 3, 4, 5]]), torch.tensor([[1, 2, 3, 4, 9]])
        for field, value in EXPECTED_CHANGE.values():
            model = tiny_model(**{field: 4 if field == "d_ff" else value})
            with torch.no_grad():
                torch.testing.assert_close(model(first)[:, :-1], model(second)[:, :-1])

    def test_no_pos_enc_cannot_distinguish_repeated_tokens(self):
        """Without positions, a run of identical tokens yields identical outputs at every position."""
        model = tiny_model(pos_encoding="none")
        learned = tiny_model(pos_encoding="learned")
        tokens = torch.tensor([[7, 7, 7, 7]])
        with torch.no_grad():
            # Causal attention over identical tokens averages identical values: every position matches.
            out = model(tokens)
            torch.testing.assert_close(out[:, 0], out[:, -1])
            self.assertFalse(torch.allclose(learned(tokens)[:, 0], learned(tokens)[:, -1]))

    def test_lr_schedule_warms_up_then_decays_to_floor(self):
        kw = dict(warmup_steps=10, total_steps=110, scheduler="cosine", eta_min_ratio=0.1)
        self.assertAlmostEqual(lr_multiplier(0, **kw), 0.1)
        self.assertAlmostEqual(lr_multiplier(9, **kw), 1.0)
        self.assertAlmostEqual(lr_multiplier(10, **kw), 1.0)
        self.assertAlmostEqual(lr_multiplier(60, **kw), 0.55)
        self.assertAlmostEqual(lr_multiplier(110, **kw), 0.1)

    def test_holm_adjustment_is_step_down_and_monotone(self):
        # p sorted: 0.01*3=0.03, 0.02*2=0.04, 0.04*1=0.04 (monotone), restored to input order.
        self.assertEqual([round(p, 6) for p in holm([0.04, 0.01, 0.02])], [0.04, 0.03, 0.04])
        self.assertEqual(holm([0.5, 0.9]), [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
