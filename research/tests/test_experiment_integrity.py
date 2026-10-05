import unittest

import torch
from omegaconf import OmegaConf

from research.experiments.train import resolved_model_config
from research.models.ffn.feed_forward import FeedForward
from research.models.transformer.transformer import TransformerLM


class ExperimentIntegrityTests(unittest.TestCase):
    def test_declared_override_changes_resolved_model(self):
        cfg = OmegaConf.create({
            "model": {"pos_encoding": "learned", "n_heads": 8},
            "experiment": {"model_overrides": {"pos_encoding": "none", "n_heads": 1}},
        })
        model = resolved_model_config(cfg)
        self.assertEqual(model.pos_encoding, "none")
        self.assertEqual(model.n_heads, 1)

    def test_linear_ffn_is_a_valid_shape_preserving_ablation(self):
        layer = FeedForward(d_model=8, d_ff=16, dropout=0.0, activation="linear")
        self.assertEqual(tuple(layer(torch.randn(2, 3, 8)).shape), (2, 3, 8))

    def test_causal_mask_prevents_future_token_leakage(self):
        torch.manual_seed(0)
        model = TransformerLM(vocab_size=32, d_model=8, n_heads=2, n_layers=1, d_ff=16, dropout=0.0, emb_dropout=0.0, attn_dropout=0.0, ff_dropout=0.0)
        model.eval()
        first = torch.tensor([[1, 2, 3, 4]])
        second = torch.tensor([[1, 2, 3, 9]])
        with torch.no_grad():
            first_logits = model(first)
            second_logits = model(second)
        torch.testing.assert_close(first_logits[:, :-1], second_logits[:, :-1])


if __name__ == "__main__":
    unittest.main()
