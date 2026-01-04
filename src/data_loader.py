from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Iterator, List, Tuple, Sequence
import torch
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

from src.utils import normalize_khmer
from config import SPLIT, SEQ_LEN, BATCH_SIZE, NUM_WORKERS

def find_raw_files(data_dir: Path, pattern: str = "*.txt") -> List[Path]:
    files = sorted([p for p in data_dir.glob(pattern) if p.is_file()])
    return files

def iter_lines(paths: Sequence[Path]) -> Iterator[str]:
    for path in paths:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                yield line

def build_clean_corpus(out_path: Path, paths: Sequence[Path], min_len: int = 1) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept, dropped = 0, 0
    with out_path.open("w", encoding="utf-8") as w:
        for line in tqdm(iter_lines(paths), desc="Normalizing (stream)"):
            s = normalize_khmer(line)
            if len(s) >= min_len:
                w.write(s + "\n")
                kept += 1
            else:
                dropped += 1
    print(f"Done. kept={kept:,} dropped={dropped:,} -> {out_path}")

# -------- deterministic split by line hash (notebook style) --------
def split_of_line(line: str, split=SPLIT) -> str:
    h = hashlib.md5(line.encode("utf-8", errors="ignore")).hexdigest()
    x = int(h[:8], 16) / 0xFFFFFFFF
    if x < split["train"]:
        return "train"
    elif x < split["train"] + split["val"]:
        return "val"
    else:
        return "test"

def iter_split_lines(clean_path: Path, which: str) -> Iterator[str]:
    with clean_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if split_of_line(s) == which:
                yield s

def chunk_tokens(token_ids: List[int], seq_len: int) -> Iterator[Tuple[List[int], List[int]]]:
    # yields (x, y) blocks with teacher forcing shift
    if len(token_ids) < 2:
        return
    for i in range(0, len(token_ids) - 1, seq_len):
        block = token_ids[i : i + seq_len + 1]
        if len(block) < 2:
            continue
        yield block[:-1], block[1:]

class StreamLMDataset(IterableDataset):
    def __init__(self, clean_path: Path, sp, split_name: str, seq_len: int):
        super().__init__()
        self.clean_path = clean_path
        self.sp = sp
        self.split_name = split_name
        self.seq_len = seq_len

    def __iter__(self):
        for line in iter_split_lines(self.clean_path, self.split_name):
            ids = [self.sp.bos_id()] + self.sp.encode(line, out_type=int) + [self.sp.eos_id()]
            for x, y in chunk_tokens(ids, self.seq_len):
                yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def collate_pad(batch, pad_id: int):
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    x_pad = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
    y_pad = torch.full((len(ys), max_len), pad_id, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        x_pad[i, : x.size(0)] = x
        y_pad[i, : y.size(0)] = y
    return x_pad, y_pad

def make_loaders(clean_txt: Path, sp, seq_len: int = SEQ_LEN, batch_size: int = BATCH_SIZE):
    pad_id = sp.pad_id()
    train_ds = StreamLMDataset(clean_txt, sp, "train", seq_len=seq_len)
    val_ds   = StreamLMDataset(clean_txt, sp, "val",   seq_len=seq_len)
    test_ds  = StreamLMDataset(clean_txt, sp, "test",  seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=lambda b: collate_pad(b, pad_id),
                              num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, collate_fn=lambda b: collate_pad(b, pad_id),
                              num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, collate_fn=lambda b: collate_pad(b, pad_id),
                              num_workers=NUM_WORKERS)
    return train_loader, val_loader, test_loader
