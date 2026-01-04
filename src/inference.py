from __future__ import annotations
import torch
from src.utils import normalize_khmer

def suggest_next(model, sp, device, text: str, topk: int = 10, temperature: float = 1.0):
    model.eval()
    s = normalize_khmer(text)
    ids = [sp.bos_id()] + sp.encode(s, out_type=int)
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(x)
        last = logits[0, -1] / max(temperature, 1e-6)
        probs = torch.softmax(last, dim=-1)
        top = torch.topk(probs, k=topk)
        out = []
        for pid, p in zip(top.indices.tolist(), top.values.tolist()):
            out.append((pid, sp.id_to_piece(pid), sp.decode([pid]), float(p)))
    return s, out

def generate_text(model, sp, device, prompt: str, max_new_tokens: int = 20, temperature: float = 1.0, topk: int = 20):
    model.eval()
    s = normalize_khmer(prompt)
    ids = [sp.bos_id()] + sp.encode(s, out_type=int)
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(x)
            last = logits[0, -1] / max(temperature, 1e-6)
            probs = torch.softmax(last, dim=-1)
            top = torch.topk(probs, k=topk)
            pid = top.indices[torch.multinomial(top.values, 1)].item()
            x = torch.cat([x, torch.tensor([[pid]], device=device)], dim=1)

    return sp.decode(x[0].tolist())
