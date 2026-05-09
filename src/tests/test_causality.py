import torch
import unittest
from src.modeling.transformer import TransformerModel

class TestCausality(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.embed_size = 64
        self.num_layers = 1
        self.num_heads = 2
        self.ff_dim = 128
        self.block_size = 32
        self.dropout = 0.0
        
        self.model = TransformerModel(
            vocab_size=self.vocab_size,
            embed_size=self.embed_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            block_size=self.block_size,
            dropout=self.dropout
        ).eval()

    def test_causality(self):
        """
        Changing a token at position t should NOT affect the logits at positions < t.
        """
        seq_len = 10
        x1 = torch.randint(0, self.vocab_size, (1, seq_len))
        
        # Create x2 identical to x1 up to position 4, but different at position 5
        x2 = x1.clone()
        x2[0, 5] = (x1[0, 5] + 1) % self.vocab_size
        
        with torch.no_grad():
            logits1, _ = self.model(x1)
            logits2, _ = self.model(x2)
        
        # Logits at positions 0 to 4 should be identical
        torch.testing.assert_close(logits1[:, :5, :], logits2[:, :5, :], rtol=1e-5, atol=1e-5)
        
        # Logits at position 5 onwards can (and likely will) differ
        # (Technically, at position 5 they should also be different if the model is non-trivial)
        self.assertFalse(torch.allclose(logits1[:, 5:, :], logits2[:, 5:, :]))

if __name__ == "__main__":
    unittest.main()
