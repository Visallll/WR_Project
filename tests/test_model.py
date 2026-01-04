import torch
from src.model import GRULanguageModel

def test_forward_shape():
    vocab = 100
    pad = 0
    m = GRULanguageModel(vocab_size=vocab, emb_dim=32, hid_dim=64, n_layers=1, dropout=0.1, pad_id=pad)
    x = torch.randint(0, vocab, (2, 10))
    logits, _ = m(x)
    assert logits.shape == (2, 10, vocab)
