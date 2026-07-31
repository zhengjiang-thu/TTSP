"""Model and algorithm defaults for TTSP."""

import math
MODEL_TYPE_CONFIG = {
    "thinking": {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "gamma": 1.0,
        "reasoning_effort": "low",
    },
    "instruct": {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "gamma": 1.0,
        "reasoning_effort": "low",
    },
}


def get_sampling_params(model_name: str) -> dict:
    """Get default sampling parameters for the given model name."""
    model_type = "thinking" if "thinking" in model_name.lower() else "instruct"
    return MODEL_TYPE_CONFIG[model_type].copy()


def compute_fresh_count(
    round_idx: int,
    budget_per_round: int,
    fresh_exploration_ratio: float,
    has_evidence_ledger: bool,
) -> int:
    """Return Algorithm 1's fresh-pool size for a round."""
    # Kept in the signature for compatibility with the initial release.  The
    # paper's schedule depends on the round index, not on whether a preceding
    # ledger call happened to fail.
    del has_evidence_ledger
    if round_idx < 0:
        raise ValueError("round_idx cannot be negative")
    if budget_per_round < 1:
        raise ValueError("budget_per_round must be at least 1")
    if not 0.0 <= fresh_exploration_ratio <= 1.0:
        raise ValueError("fresh_exploration_ratio must be in [0, 1]")
    if round_idx == 0:
        return budget_per_round
    return min(
        budget_per_round,
        math.ceil(fresh_exploration_ratio * budget_per_round),
    )
