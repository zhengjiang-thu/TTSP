"""TTSP: Test-Time Scaling over Perception.

The current implementation follows the paper's three-part loop: dual-pool
coverage, critical-token entropy selection, and Evidence-Ledger utilization,
followed by reliability-weighted aggregation over all retained traces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .outputs import PerRoundResult, TTSPOutput
    from .pipeline import TTSPPipeline


__version__ = "2.0.0"
__author__ = "Zheng Jiang"
__email__ = "jz24@mails.tsinghua.edu.cn"
__all__ = ["TTSPPipeline", "TTSPOutput", "PerRoundResult"]


def __getattr__(name: str) -> Any:
    """Load GPU-heavy dependencies only when the pipeline is requested."""
    if name == "TTSPPipeline":
        from .pipeline import TTSPPipeline

        return TTSPPipeline
    if name in {"TTSPOutput", "PerRoundResult"}:
        from .outputs import PerRoundResult, TTSPOutput

        return {"TTSPOutput": TTSPOutput, "PerRoundResult": PerRoundResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
