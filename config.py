from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts")))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Corpus build ----
CLEAN_TXT = ARTIFACTS_DIR / "clean_corpus.txt"
RAW_GLOB = os.getenv("RAW_GLOB", "*.txt")  # all raw text files inside data/

# ---- SentencePiece ----
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "32000"))
SPM_CHAR_COVERAGE = float(os.getenv("SPM_CHAR_COVERAGE", "0.9995"))
SPM_MODEL_PREFIX = ARTIFACTS_DIR / f"spm_bpe_{VOCAB_SIZE}"
SPM_MODEL_PATH = SPM_MODEL_PREFIX.with_suffix(".model")
SPM_VOCAB_PATH = SPM_MODEL_PREFIX.with_suffix(".vocab")

# ---- Split (deterministic hashing by line) ----
SPLIT = {
    "train": float(os.getenv("SPLIT_TRAIN", "0.80")),
    "val":   float(os.getenv("SPLIT_VAL",   "0.10")),
    "test":  float(os.getenv("SPLIT_TEST",  "0.10")),
}

# ---- LM dataset ----
SEQ_LEN = int(os.getenv("SEQ_LEN", "128"))
MIN_LINE_LEN = int(os.getenv("MIN_LINE_LEN", "1"))

# ---- Training ----
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", "64"))
EPOCHS       = int(os.getenv("EPOCHS", "10"))
LR           = float(os.getenv("LR", "0.001"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
GRAD_CLIP    = float(os.getenv("GRAD_CLIP", "1.0"))
NUM_WORKERS  = int(os.getenv("NUM_WORKERS", "0"))

# ---- Model ----
EMB_DIM   = int(os.getenv("EMB_DIM", "256"))
HID_DIM   = int(os.getenv("HID_DIM", "512"))
N_LAYERS  = int(os.getenv("N_LAYERS", "2"))
DROPOUT   = float(os.getenv("DROPOUT", "0.2"))

# ---- Repro ----
SEED = int(os.getenv("SEED", "42"))

# ---- Checkpoints ----
BEST_GRU_PATH = ARTIFACTS_DIR / "best_gru.pt"
