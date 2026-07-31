"""Critical-token entropy scoring and round-relative gating for TTSP."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_ENTROPY_TOP_K = 256
DEFAULT_CRITICAL_POSITIONS = 128


def _logprob_value(value: Any) -> float:
    """Return a numeric log-probability from a vLLM value or plain scalar."""
    if hasattr(value, "logprob"):
        return float(value.logprob)
    if isinstance(value, dict):
        return float(value.get("logprob", -math.inf))
    return float(value)


def calculate_token_entropies(
    logprobs_data: Sequence[Any],
    k_top: int = DEFAULT_ENTROPY_TOP_K,
) -> List[Optional[float]]:
    """Compute renormalized truncated entropy at every generated position.

    ``logprobs_data`` follows vLLM's per-position representation.  Missing
    distributions are returned as ``None`` so they cannot make a trace look
    spuriously confident.  Values are sorted before truncation because vLLM's
    mapping order is not part of its public contract.
    """
    if k_top < 1:
        raise ValueError("k_top must be at least 1")

    entropies: List[Optional[float]] = []
    for entry in logprobs_data or []:
        if not entry:
            entropies.append(None)
            continue

        values = entry.values() if isinstance(entry, dict) else entry
        logps = sorted(
            (_logprob_value(value) for value in values),
            reverse=True,
        )[:k_top]
        logps = [value for value in logps if math.isfinite(value)]
        if not logps:
            entropies.append(None)
            continue

        # Subtracting the maximum keeps exponentiation stable while preserving
        # the renormalized top-k distribution.
        maximum = max(logps)
        probs = [math.exp(value - maximum) for value in logps]
        normalizer = sum(probs)
        if not math.isfinite(normalizer) or normalizer <= 0:
            entropies.append(None)
            continue
        normalized = [probability / normalizer for probability in probs]
        entropies.append(
            -sum(
                probability * math.log(probability)
                for probability in normalized
                if probability > 0
            )
        )
    return entropies


def aggregate_highest_k_entropy(
    values: Sequence[Optional[float]],
    k: int = DEFAULT_CRITICAL_POSITIONS,
) -> Optional[float]:
    """Average the ``k`` largest finite entropy values in a trace."""
    if k < 1:
        raise ValueError("k must be at least 1")
    finite = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(value)
    )
    if not finite:
        return None
    count = min(k, len(finite))
    return sum(finite[-count:]) / count


def score_trace(
    trace: Dict[str, Any],
    critical_positions: int = DEFAULT_CRITICAL_POSITIONS,
) -> float:
    """Return Eq. (2)'s reliability score (higher is more reliable)."""
    entropy = aggregate_highest_k_entropy(
        trace.get("token_entropies", []),
        critical_positions,
    )
    return -entropy if entropy is not None else -math.inf


def find_optimal_k(traces: List[Dict[str, Any]]) -> int:
    """Compatibility shim for older callers.

    The current paper fixes the number of critical positions rather than
    adapting it to trace length.
    """
    del traces
    return DEFAULT_CRITICAL_POSITIONS


def retained_trace_count(num_traces: int, filtering_ratio: float) -> int:
    """Return the exact round-relative keep count from Algorithm 1."""
    if num_traces < 0:
        raise ValueError("num_traces cannot be negative")
    if not 0.0 <= filtering_ratio <= 1.0:
        raise ValueError("filtering_ratio must be in [0, 1]")
    if num_traces == 0:
        return 0
    return math.ceil((1.0 - filtering_ratio) * num_traces)


def filter_traces(
    traces: List[Dict[str, Any]],
    filtering_ratio: float = 0.4,
    critical_positions: int = DEFAULT_CRITICAL_POSITIONS,
    *,
    k: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Keep exactly ``ceil((1-rho)K)`` traces, ranked within one round.

    Ties are resolved by original generation order, making the result stable
    and preserving the fixed compute budget.  Retained traces are returned in
    descending reliability order so the first ``m`` can be sent directly to
    the Evidence Ledger updater.
    """
    if not traces:
        return [], -math.inf
    if k is not None:
        critical_positions = k

    ranked = []
    for index, trace in enumerate(traces):
        score = score_trace(trace, critical_positions)
        trace["reliability_score"] = score
        ranked.append((index, score, trace))

    ranked.sort(key=lambda item: (-item[1], item[0]))
    keep = retained_trace_count(len(traces), filtering_ratio)
    if keep == 0:
        return [], math.inf
    retained = [item[2] for item in ranked[:keep]]
    for rank, trace in enumerate(retained, start=1):
        trace["reliability_rank"] = rank
    threshold = ranked[keep - 1][1]
    return retained, float(threshold)


def compute_trace_weight(trace: Dict[str, Any], gamma: float = 1.0) -> float:
    """Convert a reliability score to its unnormalized vote weight."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    score = float(trace.get("reliability_score", -math.inf))
    if not math.isfinite(score):
        return 0.0
    # Scores are non-positive entropies, so overflow is not expected.  Clamp
    # defensively for externally supplied scores.
    return float(math.exp(max(-745.0, min(709.0, score / gamma))))
