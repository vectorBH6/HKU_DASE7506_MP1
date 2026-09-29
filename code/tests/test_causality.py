"""Strict causality tests for models that contain GlobalMemoryTokens.

Run: python -m unittest discover -s tests -v (from the code/ directory).

These tests are stricter than the contract-level future-input test: they
perturb the very last token and require every earlier prediction to stay
bit-exact. Any module that leaks future information into earlier positions
(e.g. softmax over the time axis or a sequence-wide reduction) will fail.
"""
import importlib
import os
import sys
import unittest

import torch


CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


def _load(module_name):
    """Import student_work.<module_name> fresh to avoid state leakage."""
    full = f'student_work.{module_name}'
    if full in sys.modules:
        del sys.modules[full]
    return importlib.import_module(full)


def _build(model_module, n_global_tokens):
    cfg = dict(
        vocab=2048, width=32, heads=4, depth=2, context=256,
        n_global_tokens=n_global_tokens, dropout=0.0,
    )
    torch.manual_seed(17)
    model = model_module.build_model(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class FutureLeakageTests(unittest.TestCase):
    """Mutating the last token must not affect any earlier prediction."""

    SEQ_LEN = 16

    def _assert_strict_causality(self, model, atol=1e-6, rtol=1e-6):
        torch.manual_seed(0)
        x = torch.randint(0, 2048, (1, self.SEQ_LEN))
        changed = x.clone()
        changed[:, -1] = (changed[:, -1] + 1) % 2048

        with torch.no_grad():
            a = model.predict_log_probs(x)
            b = model.predict_log_probs(changed)

        # Every position except the last must be unaffected by mutating the last.
        torch.testing.assert_close(a[:, :-1], b[:, :-1], atol=atol, rtol=rtol)
        # Sanity: the last-position prediction must actually change, otherwise
        # the test is vacuously satisfied by a model that ignores all inputs.
        self.assertFalse(torch.allclose(a[:, -1], b[:, -1], atol=atol, rtol=rtol),
                         'Last-position prediction did not change; the test '
                         'is not exercising the future-input path.')

    def test_student_my_with_global_memory_is_causal(self):
        """The buggy GlobalMemoryTokens is expected to FAIL this test."""
        module = _load('student_my')
        if module.build_model(dict(vocab=2048, width=32, heads=4, depth=2,
                                   context=256, n_global_tokens=0)) is None:
            self.fail('build_model returned None')
        # n_global_tokens=0 -> no GlobalMemoryTokens -> trivially causal.
        model_no_mem = _build(module, n_global_tokens=0)
        self._assert_strict_causality(model_no_mem)

        # n_global_tokens>0 -> GlobalMemoryTokens is exercised. With the
        # current implementation (softmax over time + sequence-wide sum)
        # this assertion is expected to raise.
        model_with_mem = _build(module, n_global_tokens=4)
        with self.assertRaises(AssertionError,
                               msg='Expected GlobalMemoryTokens to leak future '
                                   'information, but predictions matched.'):
            self._assert_strict_causality(model_with_mem)

    def test_student_my_casual_is_causal(self):
        """The fixed GlobalMemoryTokens must satisfy strict causality."""
        module = _load('student_my_casual')
        model_no_mem = _build(module, n_global_tokens=0)
        self._assert_strict_causality(model_no_mem)

        model_with_mem = _build(module, n_global_tokens=4)
        self._assert_strict_causality(model_with_mem)


if __name__ == '__main__':
    unittest.main()
