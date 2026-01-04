from __future__ import annotations
import torch.nn as nn

class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hid_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.drop(self.emb(x))
        out, hidden = self.gru(emb, hidden)
        out = self.drop(out)
        logits = self.fc(out)
        return logits, hidden

def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
