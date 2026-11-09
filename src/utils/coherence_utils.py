"""Coherence / degeneration metrics for steered completions.

Two families, both CPU-only and deterministic:

  (A) heuristics from src/utils/filter_utils.py: emptiness/length and character repetition.
  (B) standard neural-text-degeneration metrics: distinct-n / rep-n and a self-repetition
      measure across the generations of one condition (cf. Holtzman et al., "The Curious Case
      of Neural Text Degeneration").

Nothing here calls a model, so it can be re-run cheaply with different thresholds.
"""

import re
import zlib
from typing import Dict, List, Sequence

from src.utils.filter_utils import _char_rep_ratio, _num_words, check_degenerate, check_empty_or_short

_WORD = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())


def _ngrams(toks: Sequence[str], n: int):
    return [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]


def distinct_n(toks: Sequence[str], n: int) -> float:
    """Unique n-grams / total n-grams. 1.0 = no repetition, ->0 = degenerate looping."""
    g = _ngrams(toks, n)
    return len(set(g)) / len(g) if g else 0.0


def max_ngram_repeat(toks: Sequence[str], n: int = 4) -> int:
    """Largest number of times any single n-gram occurs -- catches tight loops."""
    g = _ngrams(toks, n)
    if not g:
        return 0
    counts: Dict[tuple, int] = {}
    for x in g:
        counts[x] = counts.get(x, 0) + 1
    return max(counts.values())


def zlib_ratio(text: str) -> float:
    """Compressed size / raw size. Lower = more redundant text."""
    b = (text or "").encode("utf-8")
    return len(zlib.compress(b)) / len(b) if b else 1.0


def completion_metrics(completion: str) -> Dict[str, float]:
    """Continuous metrics for one completion. No thresholds applied here."""
    text = (completion or "").strip()
    toks = _tokens(text)
    return {
        "n_chars": float(len(text)),
        "n_words": float(_num_words(text)),
        "char_rep_ratio": _char_rep_ratio(text),
        "distinct_1": distinct_n(toks, 1),
        "distinct_2": distinct_n(toks, 2),
        "distinct_3": distinct_n(toks, 3),
        "distinct_4": distinct_n(toks, 4),
        "max_4gram_repeat": float(max_ngram_repeat(toks, 4)),
        "zlib_ratio": zlib_ratio(text),
    }


def completion_flags(completion: str, min_words: int, max_char_rep: float, min_distinct_4: float) -> Dict[str, bool]:
    """Absolute (threshold-based) degeneration flags for one completion.

    `empty_or_short` and `degenerate_repetition` are the checks of src/utils/filter_utils.py.
    """
    short = check_empty_or_short(completion, min_words=min_words)
    degen = check_degenerate(completion, max_char_rep=max_char_rep)
    d4 = distinct_n(_tokens(completion), 4)
    flags = {
        "empty_or_short": short is not None,
        "degenerate_repetition": degen is not None,
        "low_distinct_4": bool(d4 < min_distinct_4),
    }
    flags["incoherent"] = any(flags.values())
    return flags


def self_repetition(completions: Sequence[str], n: int = 4, max_pairs: int = 2000) -> float:
    """Mean pairwise 4-gram overlap across the generations of one condition.

    Catches a failure mode per-completion metrics miss: every generation collapsing to the same
    text. 0 = all different, 1 = identical. Pairs are capped for cost (50 completions = 1225
    pairs, well under the cap).
    """
    sets = [set(_ngrams(_tokens(c), n)) for c in completions]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return 0.0
    tot, k = 0.0, 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if k >= max_pairs:
                break
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            tot += inter / union if union else 0.0
            k += 1
    return tot / k if k else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (q in 0-100). Avoids a numpy dependency here."""
    v = sorted(values)
    if not v:
        return float("nan")
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * (q / 100.0)
    lo, hi = int(pos), min(int(pos) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)
