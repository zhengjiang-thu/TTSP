"""
Output dataclasses for TTSP pipeline results.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class PerRoundResult:
    """Stores results from a single TTSP iteration round."""
    round_idx: int
    traces: List[Dict[str, Any]] = field(default_factory=list)          # All traces generated
    filtered_traces: List[Dict[str, Any]] = field(default_factory=list) # Traces that passed filter
    visual_facts: Optional[str] = None   # Extracted facts (None for last round)
    voting_results: Dict[str, Any] = field(default_factory=dict)        # Cumulative voting up to this round
    total_tokens: int = 0
    generation_time: float = 0.0
    filtering_ratio: float = 0.0         # fraction filtered out


@dataclass
class TTSPOutput:
    """Aggregated output from the full TTSP pipeline."""

    # Per-round detailed results
    per_round_results: List[PerRoundResult] = field(default_factory=list)

    # Aggregated across rounds
    all_traces: List[Dict[str, Any]] = field(default_factory=list)           # raw traces from all rounds
    all_filtered_traces: List[Dict[str, Any]] = field(default_factory=list)  # filtered across rounds

    # Final voting results: {method_name: {"answer": ..., "num_votes": ...}}
    voting_results: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    total_traces_count: int = 0
    total_tokens: int = 0
    avg_tokens_per_trace: float = 0.0

    # Timing
    generation_time: float = 0.0
    extraction_time: float = 0.0
    total_time: float = 0.0

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "voting_results": self.voting_results,
            "per_round_voting": {
                f"round_{r.round_idx}": r.voting_results
                for r in self.per_round_results
            },
            "all_traces": self.all_traces,
            "per_round_facts": [r.visual_facts for r in self.per_round_results],
            "total_traces_count": self.total_traces_count,
            "token_stats": {
                "total_tokens": self.total_tokens,
                "avg_tokens_per_trace": self.avg_tokens_per_trace,
            },
            "timing_stats": {
                "generation_time": self.generation_time,
                "extraction_time": self.extraction_time,
                "total_time": self.total_time,
            },
            "config": self.config,
            "timestamp": self.timestamp,
        }

    def to_debug_dict(self) -> Dict[str, Any]:
        """Export full debug info including generated texts and knowledge."""
        return {
            "voting_results": self.voting_results,
            "per_round_results": [
                {
                    "round_idx": r.round_idx,
                    "visual_facts": r.visual_facts,
                    "traces": [
                        {
                            "trace_id": t.get("trace_id"),
                            "round_idx": t.get("round_idx"),
                            "texts": t.get("texts"),  # Full generated text per turn
                            "extracted_answer": t.get("extracted_answer"),
                            "num_tokens": t.get("num_tokens"),
                            "num_turns": t.get("num_turns"),
                            "visual_facts_input": t.get("visual_facts_input"),
                        }
                        for t in r.traces
                    ],
                    "filtered_trace_ids": [t.get("trace_id") for t in r.filtered_traces],
                    "total_tokens": r.total_tokens,
                    "generation_time": r.generation_time,
                    "filtering_ratio": r.filtering_ratio,
                }
                for r in self.per_round_results
            ],
            "all_traces": [
                {
                    "trace_id": t.get("trace_id"),
                    "round_idx": t.get("round_idx"),
                    "texts": t.get("texts"),
                    "extracted_answer": t.get("extracted_answer"),
                    "num_tokens": t.get("num_tokens"),
                    "visual_facts_input": t.get("visual_facts_input"),
                }
                for t in self.all_traces
            ],
            "config": self.config,
            "timestamp": self.timestamp,
        }

    def print_summary(self):
        print("\n=== TTSP Summary ===")
        print(f"Rounds: {len(self.per_round_results)}")
        print(f"Total traces: {self.total_traces_count}")
        print(f"Total tokens: {self.total_tokens}")
        if self.generation_time > 0:
            print(f"Generation time: {self.generation_time:.2f}s")
        print(f"Total time: {self.total_time:.2f}s")

        if len(self.per_round_results) > 1:
            print("\n=== Per-Round Voting (cumulative) ===")
            for r in self.per_round_results:
                if r.voting_results:
                    res = r.voting_results.get("TTSP")
                    if res and res.get("answer"):
                        print(f"  Round {r.round_idx}: TTSP={res['answer']}")

        if self.voting_results:
            print("\n=== Final Voting Result ===")
            res = self.voting_results.get("TTSP")
            if res and res.get("answer"):
                print(f"  TTSP: {res['answer']} [{res.get('num_votes', 0)} votes]")
