"""
Voting strategies for TTSP final answer aggregation.
"""
from collections import Counter, defaultdict
from typing import List, Optional, Dict, Any
import numpy as np

from .reliability import filter_traces, compute_trace_weight


def simple_majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]


def weighted_majority_vote(answers: List[str], weights: List[float]) -> Optional[str]:
    if not answers:
        return None
    totals: Dict[str, float] = defaultdict(float)
    for a, w in zip(answers, weights):
        if a is not None:
            totals[str(a)] += float(w)
    return max(totals, key=lambda x: totals[x]) if totals else None


def compute_voting_results(
    traces: List[Dict[str, Any]],
    filtering_ratio: float = 0.4,
    gamma: float = 0.1,
) -> Dict[str, Any]:
    """
    Compute TTSP voting result using reliability-filtered weighted vote.
    
    Args:
        traces: List of perception traces with extracted answers and reliability scores.
        filtering_ratio: Fraction of low-reliability traces to discard.
        gamma: Temperature for reliability-weighted voting.
        
    Returns:
        Dict containing TTSP voting result with answer, num_votes, and threshold.
    """
    if not traces:
        return {"TTSP": None}

    answers = [t.get('extracted_answer') for t in traces]

    # TTSP: filter then weight
    filtered, threshold = filter_traces(traces, filtering_ratio=filtering_ratio)
    if filtered:
        f_answers = [t.get('extracted_answer') for t in filtered]
        weights = [compute_trace_weight(t, gamma) for t in filtered]
        ttsp_answer = weighted_majority_vote(
            [a for a in f_answers if a],
            [w for a, w in zip(f_answers, weights) if a]
        )
        num_votes = sum(1 for a in f_answers if a)
    else:
        # Fallback to simple majority if all traces are filtered out
        ttsp_answer = simple_majority_vote(answers)
        num_votes = len([a for a in answers if a])
        threshold = 0.0

    return {
        "TTSP": {
            "answer": ttsp_answer,
            "num_votes": num_votes,
            "threshold": threshold,
        },
    }
