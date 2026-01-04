from __future__ import annotations
import math
import time
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.model import GRULanguageModel, count_params
from config import (
    EMB_DIM, HID_DIM, N_LAYERS, DROPOUT,
    EPOCHS, LR, WEIGHT_DECAY, GRAD_CLIP,
    BEST_GRU_PATH
)

@dataclass
class Metrics:
    loss: float
    ppl: float
    top1: float
    top3: float

def batch_metrics(logits: torch.Tensor, targets: torch.Tensor, pad_id: int, k3: int = 3) -> Metrics:
    """Compute CE loss + perplexity + top-1/top-3 accuracy ignoring pad tokens."""
    B, T, V = logits.shape
    logits2 = logits.reshape(B * T, V)
    targets2 = targets.reshape(B * T)

    mask = targets2 != pad_id
    if mask.sum().item() == 0:
        return Metrics(loss=0.0, ppl=float("inf"), top1=0.0, top3=0.0)

    logits_m = logits2[mask]
    targets_m = targets2[mask]

    loss = F.cross_entropy(logits_m, targets_m, reduction="mean").item()

    # top-k
    with torch.no_grad():
        top1 = (logits_m.argmax(dim=-1) == targets_m).float().mean().item()
        topk = torch.topk(logits_m, k=min(k3, logits_m.size(-1)), dim=-1).indices
        top3 = (topk == targets_m.unsqueeze(-1)).any(dim=-1).float().mean().item()

    ppl = math.exp(loss) if loss < 50 else float("inf")
    return Metrics(loss=loss, ppl=ppl, top1=top1, top3=top3)

def run_epoch(model, loader, loss_fn, opt, scaler, device, pad_id: int, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0
    sum_top1 = 0.0
    sum_top3 = 0.0

    t0 = time.time()
    pbar = tqdm(loader, desc="train" if train else "eval")
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits, _ = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt)
            scaler.update()

        # metrics on unpadded tokens
        with torch.no_grad():
            m = batch_metrics(logits.detach(), y, pad_id=pad_id, k3=3)
            # count tokens (non-pad)
            tokens = (y != pad_id).sum().item()
            total_tokens += tokens
            total_loss += m.loss * tokens
            sum_top1 += m.top1 * tokens
            sum_top3 += m.top3 * tokens

        avg_loss = total_loss / max(total_tokens, 1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{(sum_top1/max(total_tokens,1)):.3f}")

    dt = max(time.time() - t0, 1e-9)
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    top1 = sum_top1 / max(total_tokens, 1)
    top3 = sum_top3 / max(total_tokens, 1)
    tokens_per_sec = total_tokens / dt

    return Metrics(loss=avg_loss, ppl=ppl, top1=top1, top3=top3), tokens_per_sec

@torch.no_grad()
def evaluate(model, loader, device, pad_id: int) -> Metrics:
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="mean")
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    sum_top1 = 0.0
    sum_top3 = 0.0

    for x, y in tqdm(loader, desc="test"):
        x = x.to(device)
        y = y.to(device)
        logits, _ = model(x)
        m = batch_metrics(logits, y, pad_id=pad_id, k3=3)
        tokens = (y != pad_id).sum().item()
        total_tokens += tokens
        total_loss += m.loss * tokens
        sum_top1 += m.top1 * tokens
        sum_top3 += m.top3 * tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return Metrics(loss=avg_loss, ppl=ppl, top1=sum_top1/max(total_tokens,1), top3=sum_top3/max(total_tokens,1))

def train_gru(train_loader, val_loader, device, vocab_size: int, pad_id: int, save_path: Path = BEST_GRU_PATH):
    model = GRULanguageModel(
        vocab_size=vocab_size,
        emb_dim=EMB_DIM,
        hid_dim=HID_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        pad_id=pad_id
    ).to(device)

    print("GRU params:", count_params(model))

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="mean")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_ppl": [],
        "val_top1": [],
        "val_top3": [],
        "tokens_per_sec": [],
        "gpu_peak_gb": [],
    }

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_m, tps = run_epoch(model, train_loader, loss_fn, opt, scaler, device, pad_id, train=True)
        val_m, _ = run_epoch(model, val_loader, loss_fn, opt, scaler, device, pad_id, train=False)

        scheduler.step(val_m.loss)

        history["train_loss"].append(float(train_m.loss))
        history["val_loss"].append(float(val_m.loss))
        history["val_ppl"].append(float(val_m.ppl))
        history["val_top1"].append(float(val_m.top1))
        history["val_top3"].append(float(val_m.top3))
        history["tokens_per_sec"].append(float(tps))

        peak_gb = 0.0
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        history["gpu_peak_gb"].append(float(peak_gb))

        print(f"train loss={train_m.loss:.4f} ppl={train_m.ppl:.2f} top1={train_m.top1:.3f} top3={train_m.top3:.3f}")
        print(f"  val loss={val_m.loss:.4f} ppl={val_m.ppl:.2f} top1={val_m.top1:.3f} top3={val_m.top3:.3f}")
        print(f"  tokens/sec={tps:,.0f} peak_gpu_gb={peak_gb:.2f}")

        if val_m.loss < best_val:
            best_val = val_m.loss
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "vocab_size": vocab_size,
                        "emb_dim": EMB_DIM,
                        "hid_dim": HID_DIM,
                        "n_layers": N_LAYERS,
                        "dropout": DROPOUT,
                    },
                },
                save_path,
            )
            print("Saved best:", save_path)

    return model, history, save_path

def load_best_gru(path: Path, device, pad_id: int) -> GRULanguageModel:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = GRULanguageModel(
        vocab_size=cfg["vocab_size"],
        emb_dim=cfg["emb_dim"],
        hid_dim=cfg["hid_dim"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        pad_id=pad_id
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model
