import torch
import torch.nn.functional as F
from .model_loader import model, tokenizer, device

MAX_LENGTH = 64  # SAME as training

def predict_top_k(text: str, k: int = 5):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # last token logits
    logits = outputs.logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)

    top_probs, top_ids = torch.topk(probs, k)

    results = []
    for prob, idx in zip(top_probs[0], top_ids[0]):
        token = tokenizer.decode([idx]).strip()
        results.append({
            "word": token,
            "confidence": round(float(prob), 4)
        })

    return results
