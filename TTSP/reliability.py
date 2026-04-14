"""
Reliability scoring and filtering for TTSP.

Reliability is measured as negative token-level entropy: high-entropy tokens
indicate uncertain generation (noisy thinking), low-entropy tokens indicate
confident generation. We filter out traces whose top-k mean entropy exceeds
the population threshold.
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict


# ============= TOKEN-LEVEL ENTROPY =============

def calculate_token_entropies(logprobs_data: List[Any], k: int = 10) -> List[float]:
    """Compute per-token entropy: -sum(p*log(p)) normalized over top-k logprobs."""
    entropies = []
    for entry in (logprobs_data or []):
        if not entry:
            entropies.append(0.0)
            continue
        vals = entry.values() if isinstance(entry, dict) else entry
        probs = np.array([math.exp(getattr(v, 'logprob', v)) for v in vals][:k])
        probs = probs[probs > 0]
        if len(probs) > 0:
            probs /= probs.sum()
            entropies.append(round(float(-np.sum(probs * np.log(probs))), 6))
        else:
            entropies.append(0.0)
    return entropies


def aggregate_highest_k_entropy(values: List[float], k: int) -> float:
    """Mean of the top-k highest entropy values (measures worst-case uncertainty)."""
    if not values:
        return 0.0
    num = min(k, len(values))
    return float(np.mean(np.partition(values, -num)[-num:])) if num > 0 else 0.0


# ============= TRACE-LEVEL SCORING =============

def score_trace(trace: Dict[str, Any], k: int = 32) -> float:
    """
    Compute reliability score for a trace as negative mean top-k entropy.
    Higher score = more reliable (less uncertain).
    """
    entropies = trace.get('token_entropies', [])
    if not entropies:
        return -float('inf')
    return -aggregate_highest_k_entropy(entropies, k)


def find_optimal_k(traces: List[Dict[str, Any]]) -> int:
    """Estimate optimal k as ~20% of mean trace length, bounded to [8, 64]."""
    if not traces:
        return 32
    lengths = [len(t.get('token_entropies', [])) for t in traces if t.get('token_entropies')]
    if not lengths:
        return 32
    mean_len = np.mean(lengths)
    k = int(0.2 * mean_len)
    return max(8, min(64, k))


# ============= FILTERING =============

def filter_traces(
    traces: List[Dict[str, Any]],
    filtering_ratio: float = 0.4,
    k: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Keep the top (1 - filtering_ratio) fraction of traces by reliability score.

    Returns:
        (filtered_traces, threshold) - traces that passed and the score threshold used.
    """
    if not traces:
        return [], 0.0

    optimal_k = k if k is not None else find_optimal_k(traces)
    scores = [score_trace(t, optimal_k) for t in traces]

    # Threshold: keep top (1 - filtering_ratio) by score
    threshold = float(np.percentile(scores, filtering_ratio * 100))

    filtered = [t for t, s in zip(traces, scores) if s >= threshold]

    # Attach scores for downstream weighting
    for t, s in zip(traces, scores):
        t['reliability_score'] = s

    # Fallback: if nothing passes, return all
    return filtered if filtered else traces, threshold


# ============= WEIGHTED SCORING =============

def compute_trace_weight(trace: Dict[str, Any], gamma: float = 0.1) -> float:
    """
    Convert reliability score to a non-negative weight.
    weight = exp(score / gamma), where score = -mean_top_k_entropy.
    """
    score = trace.get('reliability_score', 0.0)
    if score == -float('inf') or math.isnan(score):
        return 0.0
    epsilon = 1e-6
    return float(math.exp(score / (abs(gamma) + epsilon)))
