\
    from __future__ import annotations
    import random
    import re
    import unicodedata
    from typing import Iterable
    import numpy as np
    import torch

    ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff"}
    KHMER_DIGITS = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")

    # Khmer-only allowed sets (no A-Za-z)
    # Keep Khmer ranges + digits + whitespace + punctuation
    ALLOWED_RE = re.compile(
        r"[^0-9"
        r"\u1780-\u17FF"   # Khmer
        r"\u19E0-\u19FF"   # Khmer symbols
        r"\s"
        r"\.\,\!\?\:\;\-\(\)\"\'"
        r"\u2000-\u206F"   # General punctuation
        r"\u3000-\u303F"   # CJK punctuation
        r"\u00A0"
        r"]+"
    )
    MULTISPACE_RE = re.compile(r"\s+")

    def normalize_khmer(text: str) -> str:
        \"\"\"Notebook-equivalent Khmer normalization.
        - NFC normalize
        - remove zero-width chars
        - convert Khmer digits to Latin digits
        - remove non-Khmer/invalid chars (Latin letters, etc.)
        - collapse whitespace
        \"\"\"
        if not text:
            return ""
        # Unicode normalize
        text = unicodedata.normalize("NFC", text)
        # remove zero-width
        text = "".join(ch for ch in text if ch not in ZERO_WIDTH)
        # translate Khmer digits -> Latin
        text = text.translate(KHMER_DIGITS)
        # filter to allowed chars
        text = ALLOWED_RE.sub(" ", text)
        # collapse spaces
        text = MULTISPACE_RE.sub(" ", text).strip()
        return text

    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def device_from_arg(device: str | None = None) -> torch.device:
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
