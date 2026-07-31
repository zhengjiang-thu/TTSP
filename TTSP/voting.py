"""Reliability-weighted answer aggregation for retained TTSP traces."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple


def normalize_answer(answer: Any) -> str:
    """Map superficial textual variants to a stable answer class."""
    if answer is None:
        return ""
    value = unicodedata.normalize("NFKC", str(answer)).strip()
    boxed = re.fullmatch(r"\\?boxed\{(.*)\}", value, flags=re.DOTALL)
    if boxed:
        value = boxed.group(1).strip()
    value = re.sub(r"\s+", " ", value).strip(" \t\n\r.,;:!?\"'()[]")
    if re.fullmatch(r"[A-Za-z]", value):
        return value.upper()
    return value.casefold()


def _answer_class(answer: Any, options: Optional[Sequence[str]]) -> Tuple[str, str]:
    """Return (class key, display answer), matching option text when possible."""
    normalized = normalize_answer(answer)
    if not normalized:
        return "", ""

    if options:
        for index, option in enumerate(options):
            letter = chr(65 + index)
            if normalized in {letter, letter.casefold()}:
                return letter, letter
            if normalized == normalize_answer(option):
                return letter, letter

    display = str(answer).strip()
    if re.fullmatch(r"[A-Za-z]", display):
        display = display.upper()
    return normalized, display


def simple_majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    valid = [answer for answer in answers if normalize_answer(answer)]
    if not valid:
        return None
    counts = Counter(normalize_answer(answer) for answer in valid)
    winner = counts.most_common(1)[0][0]
    return next(str(answer).strip() for answer in valid if normalize_answer(answer) == winner)


def weighted_majority_vote(answers: List[str], weights: List[float]) -> Optional[str]:
    if not answers:
        return None
    totals: Dict[str, float] = defaultdict(float)
    displays: Dict[str, str] = {}
    for answer, weight in zip(answers, weights):
        key = normalize_answer(answer)
        if key:
            totals[key] += float(weight)
            displays.setdefault(key, str(answer).strip())
    winner = max(totals, key=totals.get) if totals else None
    return displays.get(winner) if winner else None


def compute_voting_results(
    retained_traces: List[Dict[str, Any]],
    gamma: float = 1.0,
    options: Optional[Sequence[str]] = None,
    **legacy_kwargs: Any,
) -> Dict[str, Any]:
    """Vote over traces already retained by per-round entropy gating.

    Algorithm 1 gates each round exactly once and accumulates the survivors in
    ``F``.  This function therefore never re-filters the union across rounds.
    ``filtering_ratio`` is accepted and ignored only for source compatibility
    with the initial public release.
    """
    legacy_kwargs.pop("filtering_ratio", None)
    if legacy_kwargs:
        unknown = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"unexpected voting arguments: {unknown}")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    valid = []
    for trace in retained_traces:
        key, display = _answer_class(trace.get("extracted_answer"), options)
        score = float(trace.get("reliability_score", -math.inf))
        if key and math.isfinite(score):
            valid.append((trace, key, display, score))

    if not valid:
        return {
            "TTSP": {
                "answer": None,
                "num_votes": 0,
                "threshold": None,
                "weight_totals": {},
            }
        }

    # Subtracting the best score leaves Eq. (3)'s argmax unchanged and avoids
    # underflow for small gamma.
    best_score = max(item[3] for item in valid)
    totals: Dict[str, float] = defaultdict(float)
    displays: Dict[str, str] = {}
    counts: Dict[str, int] = defaultdict(int)
    for _, key, display, score in valid:
        weight = math.exp((score - best_score) / gamma)
        totals[key] += weight
        counts[key] += 1
        displays.setdefault(key, display)

    winner = max(totals, key=lambda key: (totals[key], counts[key], key))
    return {
        "TTSP": {
            "answer": displays[winner],
            "num_votes": len(valid),
            "winning_votes": counts[winner],
            # Gating is round-relative now, so there is no final global threshold.
            "threshold": None,
            "weight_totals": dict(totals),
        }
    }
