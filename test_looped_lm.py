import unittest

import torch

from looped_lm import LoopedLMConfig, LoopedLanguageModel


class LoopedLMTest(unittest.TestCase):
    def test_forward_and_generation(self) -> None:
        model = LoopedLanguageModel(LoopedLMConfig(vocab_size=23, block_size=8, d_model=32, n_heads=4))
        tokens = torch.randint(23, (2, 8))
        logits, loss = model(tokens, tokens, loops=2)
        self.assertEqual(logits.shape, (2, 8, 23))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.generate(tokens[:, :3], 4, loops=3).shape, (2, 7))


    def test_core_is_shared(self) -> None:
        model = LoopedLanguageModel(LoopedLMConfig(vocab_size=11, d_model=32, n_heads=4, num_loops=6))
        names = [name for name, _ in model.named_parameters()]
        self.assertTrue(any(name.startswith("core.") for name in names))
        self.assertFalse(any("core.0" in name or "core.1" in name for name in names))


if __name__ == "__main__":
    unittest.main()
