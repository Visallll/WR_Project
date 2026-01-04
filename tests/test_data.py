from src.utils import normalize_khmer

def test_normalize_khmer_removes_latin():
    s = "hello សួស្តី world"
    out = normalize_khmer(s)
    assert "hello" not in out and "world" not in out
