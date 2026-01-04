from __future__ import annotations
import argparse
from pathlib import Path

import sentencepiece as spm

from config import (
    DATA_DIR, ARTIFACTS_DIR, CLEAN_TXT, RAW_GLOB,
    SPM_MODEL_PATH, SPM_MODEL_PREFIX, VOCAB_SIZE, SPM_CHAR_COVERAGE,
    SEQ_LEN, BATCH_SIZE, MIN_LINE_LEN, BEST_GRU_PATH, SEED
)
from src.utils import set_seed, device_from_arg, normalize_khmer
from src.data_loader import find_raw_files, build_clean_corpus, make_loaders
from src.trainer import train_gru, load_best_gru, evaluate
from src.inference import suggest_next, generate_text

def train_sentencepiece(input_txt: Path, model_prefix: Path, vocab_size: int, character_coverage: float):
    spm.SentencePieceTrainer.Train(
        input=str(input_txt),
        model_prefix=str(model_prefix),
        vocab_size=int(vocab_size),
        model_type="bpe",
        character_coverage=float(character_coverage),
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        input_sentence_size=2000000,
        shuffle_input_sentence=True,
    )
    print("Saved:", model_prefix.with_suffix(".model"))
    print("Saved:", model_prefix.with_suffix(".vocab"))

def ensure_clean_corpus():
    raw_files = find_raw_files(DATA_DIR, RAW_GLOB)
    if not raw_files:
        raise FileNotFoundError(f"No raw text files found in {DATA_DIR} with pattern '{RAW_GLOB}'. Put your corpus in data/.")
    if not CLEAN_TXT.exists():
        print("Building clean corpus:", CLEAN_TXT)
        build_clean_corpus(CLEAN_TXT, raw_files, min_len=MIN_LINE_LEN)
    else:
        print("Clean corpus exists:", CLEAN_TXT)
    return raw_files

def ensure_sentencepiece():
    if not SPM_MODEL_PATH.exists():
        print("Training SentencePiece:", SPM_MODEL_PATH)
        train_sentencepiece(CLEAN_TXT, SPM_MODEL_PREFIX, vocab_size=VOCAB_SIZE, character_coverage=SPM_CHAR_COVERAGE)
    else:
        print("SentencePiece model exists:", SPM_MODEL_PATH)
    sp = spm.SentencePieceProcessor()
    sp.load(str(SPM_MODEL_PATH))
    return sp

def cmd_train(device):
    set_seed(SEED)
    ensure_clean_corpus()
    sp = ensure_sentencepiece()

    train_loader, val_loader, test_loader = make_loaders(CLEAN_TXT, sp, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    vocab_size = sp.get_piece_size()
    pad_id = sp.pad_id()

    model, history, best_path = train_gru(train_loader, val_loader, device, vocab_size=vocab_size, pad_id=pad_id, save_path=BEST_GRU_PATH)

    best = load_best_gru(best_path, device, pad_id=pad_id)
    test_m = evaluate(best, test_loader, device, pad_id=pad_id)
    print("TEST:", test_m)

    # save history for plotting later
    hist_path = ARTIFACTS_DIR / "history.json"
    hist_path.write_text(__import__("json").dumps(history, indent=2), encoding="utf-8")
    print("Saved history:", hist_path)

def cmd_demo(device, text: str):
    sp = ensure_sentencepiece()
    if not BEST_GRU_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {BEST_GRU_PATH}. Run: python main.py --train")
    model = load_best_gru(BEST_GRU_PATH, device, pad_id=sp.pad_id())

    norm, sug = suggest_next(model, sp, device, text, topk=10, temperature=1.0)
    print("Prompt:", norm)
    for pid, piece, decoded, p in sug:
        print(f"{p:.4f} | id={pid:5d} | piece={piece} | decoded='{decoded}'")

    print("\nGenerated:")
    print(generate_text(model, sp, device, text, max_new_tokens=30, temperature=1.0, topk=20))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="Run full training pipeline")
    ap.add_argument("--demo", type=str, default=None, help="Run inference demo on a prompt")
    ap.add_argument("--device", type=str, default=None, help="Force device, e.g. cuda, cpu")
    args = ap.parse_args()

    device = device_from_arg(args.device)
    print("Device:", device)
    print("Data dir:", DATA_DIR)
    print("Artifacts:", ARTIFACTS_DIR)

    if args.train:
        cmd_train(device)
    if args.demo is not None:
        cmd_demo(device, args.demo)
    if (not args.train) and (args.demo is None):
        ap.print_help()

if __name__ == "__main__":
    main()
