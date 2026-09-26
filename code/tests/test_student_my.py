"""Regression tests for student_my model configurations, precision and checkpoints."""
import io
import unittest

import torch
from torch.nn import functional as F

from student_my import RMSNorm, RotaryPositionalEmbedding, build_model


class StudentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        self.config = dict(vocab=64, width=32, heads=4, depth=2, context=16)

    def test_small_and_unusual_head_counts(self):
        # Covers odd partial-RoPE dimensions, tiny heads and a nondivisible
        # heads // 4 default (14 // 4 == 3, but 3 does not divide 14).
        for width, heads in ((40, 4), (12, 4), (8, 4), (112, 14)):
            with self.subTest(width=width, heads=heads):
                model = build_model(self.config | dict(width=width, heads=heads)).eval()
                logits = model(torch.randint(64, (2, 7)))
                self.assertEqual(tuple(logits.shape), (2, 7, 64))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertEqual(heads % model.blocks[0].attn.n_kv_heads, 0)

    def test_explicit_architecture_options(self):
        for kv_heads in (1, 2, 4):
            with self.subTest(kv_heads=kv_heads):
                model = build_model(self.config | dict(
                    kv_heads=kv_heads, hidden_dim=48, dropout=0.0,
                    rope_fraction=1.0, rope_base=5000.0))
                attn = model.blocks[0].attn
                self.assertEqual(attn.n_kv_heads, kv_heads)
                self.assertEqual(attn.rope.rope_dim, 8)
                self.assertEqual(model.blocks[0].ffn.w1.out_features, 48)
                self.assertEqual(attn.drop.p, 0.0)
                self.assertTrue(torch.isfinite(model(torch.randint(64, (2, 7)))).all())

    def test_invalid_configs_fail_early(self):
        changes = [dict(width=31), dict(width=4), dict(heads=0), dict(depth=0),
                   dict(vocab=-1), dict(context=0), dict(width=True),
                   dict(kv_heads=3), dict(kv_heads=0), dict(kv_heads=1.5),
                   dict(hidden_dim=0), dict(dropout=-0.1), dict(dropout=1.0),
                   dict(dropout=float('nan')), dict(rope_fraction=0),
                   dict(rope_fraction=1.1), dict(rope_base=1),
                   dict(rope_base=float('inf')), dict(rope_base='10000'),
                   dict(rope_fraction=None), dict(rope_fraction=True),
                   dict(dropout=False), dict(residual_dropout=-0.1),
                   dict(residual_dropout=1.0), dict(residual_dropout=True),
                   dict(residual_dropout=float('nan'))]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_model(self.config | change)
        config = dict(self.config)
        del config['vocab']
        with self.assertRaisesRegex(ValueError, 'vocab'):
            build_model(config)

    def test_input_shape_dtype_and_context(self):
        model = build_model(self.config).eval()
        for shape in ((8,), (1, 2, 3), (0, 4), (2, 0), (2, 17)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.zeros(shape, dtype=torch.long))
        with self.assertRaises(TypeError):
            model(torch.zeros(2, 8))
        for length in (1, 16):
            logp = model.predict_log_probs(torch.zeros(2, length, dtype=torch.int32))
            self.assertEqual(tuple(logp.shape), (2, length, 64))
            torch.testing.assert_close(logp.logsumexp(-1), torch.zeros(2, length),
                                       atol=1e-6, rtol=0)

    def test_residual_dropout_regularizes_training_without_changing_eval(self):
        reference = build_model(self.config | dict(dropout=0.0)).eval()
        regularized = build_model(self.config | dict(dropout=0.0, residual_dropout=0.3))
        regularized.load_state_dict(reference.state_dict(), strict=True)
        ids = torch.randint(64, (2, 9))
        regularized.eval()
        with torch.no_grad():
            torch.testing.assert_close(regularized(ids), reference(ids), atol=0, rtol=0)
        regularized.train()
        first, second = regularized(ids), regularized(ids)
        self.assertFalse(torch.equal(first, second))
        F.cross_entropy(first[:, :-1].flatten(0, 1), ids[:, 1:].flatten()).backward()
        self.assertTrue(torch.isfinite(regularized.blocks[0].ffn.w3.weight.grad).all())

    def test_rmsnorm_large_half_inputs_match_float_reference(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.tensor([[1000., -2000., 3000., -4000.]],
                                 dtype=dtype, requires_grad=True)
                norm = RMSNorm(4)
                actual = norm(x)
                values = x.detach().float()
                expected = values / torch.sqrt(values.square().mean(-1, keepdim=True) + 1e-6)
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual.float(), expected, atol=0.01, rtol=0.01)
                actual.float().sum().backward()
                self.assertTrue(torch.isfinite(x.grad).all())

    def test_rope_cache_preserves_dtype_and_positions(self):
        rope = RotaryPositionalEmbedding(10)
        for dtype, length in ((torch.float32, 16), (torch.bfloat16, 8),
                              (torch.float16, 12), (torch.float32, 4)):
            with self.subTest(dtype=dtype, length=length):
                x = torch.randn(2, 3, length, 10).to(dtype)
                rotated, _ = rope(x, x)
                expected, _ = RotaryPositionalEmbedding(10)(x, x)
                self.assertEqual(rotated.dtype, dtype)
                torch.testing.assert_close(rotated, expected)
                torch.testing.assert_close(rotated[:, :, 0], x[:, :, 0])
                torch.testing.assert_close(rotated[..., rope.rope_dim:], x[..., rope.rope_dim:])
                torch.testing.assert_close(rotated.float().square().sum(-1),
                                           x.float().square().sum(-1), atol=0.1, rtol=0.01)
        self.assertEqual(rope.state_dict(), {})

    def test_rope_rebuilds_after_module_dtype_conversion(self):
        rope = RotaryPositionalEmbedding(64)
        x = torch.randn(1, 2, 256, 64)
        rope(x, x)
        rope.half()
        actual, _ = rope(x.half(), x.half())
        expected, _ = RotaryPositionalEmbedding(64)(x.half(), x.half())
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        rope.half().float()
        actual, _ = rope(x, x)
        expected, _ = RotaryPositionalEmbedding(64)(x, x)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_bf16_autocast_backward_after_fp32_cache(self):
        model = build_model(self.config)
        ids = torch.randint(64, (2, 9))
        model(ids[:, :-1])  # Warm the positional caches in FP32 first.
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logits = model(ids[:, :-1])
            loss = F.cross_entropy(logits.float().flatten(0, 1), ids[:, 1:].flatten())
        self.assertEqual(logits.dtype, torch.bfloat16)
        loss.backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_inference_cache_can_be_reused_for_training(self):
        model = build_model(self.config).eval()
        ids = torch.randint(64, (2, 9))
        with torch.inference_mode():
            model.predict_log_probs(ids)
        model.train()
        loss = F.cross_entropy(model(ids[:, :-1]).flatten(0, 1), ids[:, 1:].flatten())
        loss.backward()
        self.assertTrue(torch.isfinite(model.blocks[0].attn.q_proj.weight.grad).all())

    def test_prefix_predictions_ignore_future_and_cache_length(self):
        model = build_model(self.config).eval()
        ids = torch.randint(64, (2, 16))
        with torch.no_grad():
            short = model.predict_log_probs(ids[:, :5])
            full = model.predict_log_probs(ids)
            again = model.predict_log_probs(ids[:, :5])
        torch.testing.assert_close(short, full[:, :5], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(short, again, atol=1e-6, rtol=1e-6)

    def test_extreme_attention_scale_stays_finite(self):
        model = build_model(self.config)
        with torch.no_grad():
            for block in model.blocks:
                block.attn.log_temperature.copy_(torch.tensor([-1000., -10., 10., 1000.]))
        ids = torch.randint(64, (2, 9))
        logits = model(ids[:, :-1])
        self.assertTrue(torch.isfinite(logits).all())
        F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_optimizer_update_and_checkpoint_roundtrip(self):
        model = build_model(self.config | dict(dropout=0.0)).train()
        ids = torch.arange(9).repeat(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        initial = F.cross_entropy(model(ids[:, :-1]).flatten(0, 1), ids[:, 1:].flatten()).item()
        for _ in range(12):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(ids[:, :-1]).flatten(0, 1), ids[:, 1:].flatten())
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = model.predict_log_probs(ids[:, :-1])
            final = F.nll_loss(expected.flatten(0, 1), ids[:, 1:].flatten()).item()
        self.assertLess(final, initial * 0.8)
        self.assertIs(model.token_emb.weight, model.lm_head.weight)
        self.assertFalse(any('cached' in key for key in model.state_dict()))

        stream = io.BytesIO()
        torch.save({'config': model.config, 'model': model.state_dict()}, stream)
        stream.seek(0)
        checkpoint = torch.load(stream, weights_only=True)
        restored = build_model(checkpoint['config']).eval()
        restored.load_state_dict(checkpoint['model'], strict=True)
        self.assertIs(restored.token_emb.weight, restored.lm_head.weight)
        with torch.no_grad():
            torch.testing.assert_close(restored.predict_log_probs(ids[:, :-1]), expected,
                                       atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
