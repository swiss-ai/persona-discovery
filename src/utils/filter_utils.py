"""Text heuristics for completions: emptiness/length and character repetition."""
from __future__ import annotations

from typing import Dict, Optional, Tuple
from collections import Counter
import re



# ---------- helpers ----------

_WORD_RE = re.compile(r"\w+", re.UNICODE)

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _num_words(s: str) -> int:
    return len(_WORD_RE.findall(_normalize(s)))

def _char_rep_ratio(s: str) -> float:
    """max character frequency / total non-whitespace chars"""
    s = s or ""
    chars = [c for c in s if not c.isspace()]
    if not chars:
        return 1.0
    c = Counter(chars)
    return c.most_common(1)[0][1] / len(chars)


# ---------- checks ----------

def check_empty_or_short(completion: str, min_words: int = 10) -> Optional[Tuple[str, Dict[str, float]]]:
    comp = (completion or "").strip()
    if not comp:
        return ("empty_completion", {"num_words": 0})
    nw = _num_words(comp)
    if nw < min_words:
        return ("too_short", {"num_words": nw})
    return None

def check_degenerate(completion: str, max_char_rep: float = 0.35, min_len: int = 50) -> Optional[Tuple[str, Dict[str, float]]]:
    comp = (completion or "").strip()
    if len(comp) < min_len:
        return None  # let short check handle it
    r = _char_rep_ratio(comp)
    if r > max_char_rep:
        return ("degenerate_repetition", {"char_rep_ratio": r})
    return None
