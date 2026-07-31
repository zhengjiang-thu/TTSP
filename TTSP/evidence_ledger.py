"""Deterministic Evidence Ledger updates for TTSP."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any, Dict, List, Optional, Sequence

from .prompts import build_evidence_ledger_messages


@dataclass(frozen=True)
class LedgerUpdate:
    """One deterministic ledger-update result and its generated-token cost."""

    text: str
    num_tokens: int = 0


def _section_items(text: str, heading: str, next_heading: Optional[str]) -> List[str]:
    start_match = re.search(rf"(?im)^##\s*{re.escape(heading)}\s*$", text)
    if not start_match:
        return []
    end = len(text)
    if next_heading:
        end_match = re.search(
            rf"(?im)^##\s*{re.escape(next_heading)}\s*$",
            text[start_match.end():],
        )
        if end_match:
            end = start_match.end() + end_match.start()
    block = text[start_match.end():end].strip()
    if not block:
        return []

    items: List[str] = []
    current: List[str] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"^(?:\d+[.)]|[-*])\s*(.+)$", line)
        if match:
            if current:
                items.append(" ".join(current))
            current = [match.group(1).strip()]
        elif current:
            current.append(line)
        else:
            current = [line]
    if current:
        items.append(" ".join(current))
    return [item for item in items if item and item.casefold() != "none identified."]


def normalize_evidence_ledger(
    text: str,
    confirmed_cap: int = 8,
    conflict_cap: int = 4,
) -> str:
    """Normalize headings and enforce the two bounded ledger tiers."""
    if confirmed_cap < 0 or conflict_cap < 0:
        raise ValueError("ledger caps cannot be negative")
    confirmed = _section_items(text or "", "CONFIRMED KNOWLEDGE", "OPEN CONFLICTS")
    conflicts = _section_items(text or "", "OPEN CONFLICTS", None)

    confirmed = confirmed[:confirmed_cap]
    conflicts = conflicts[:conflict_cap]
    confirmed_lines = (
        "\n".join(f"{index}. {item}" for index, item in enumerate(confirmed, start=1))
        if confirmed
        else "None identified."
    )
    conflict_lines = (
        "\n".join(f"{index}. {item}" for index, item in enumerate(conflicts, start=1))
        if conflicts
        else "None identified."
    )
    return (
        "## CONFIRMED KNOWLEDGE\n"
        f"{confirmed_lines}\n\n"
        "## OPEN CONFLICTS\n"
        f"{conflict_lines}"
    )


class EvidenceLedgerUpdater:
    """Update the dual-tier Evidence Ledger with one greedy model call."""

    def __init__(
        self,
        llm: Any,
        processor: Any,
        max_new_tokens: int = 1024,
        confirmed_cap: int = 8,
        conflict_cap: int = 4,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if confirmed_cap < 0 or conflict_cap < 0:
            raise ValueError("ledger caps cannot be negative")
        self.llm = llm
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.confirmed_cap = confirmed_cap
        self.conflict_cap = conflict_cap

    def _prepare(self, messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        from qwen_vl_utils import process_vision_info

        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="low",
        )
        image_inputs, video_inputs = process_vision_info(messages)
        mm_data: Dict[str, Any] = {}
        if image_inputs:
            mm_data["image"] = image_inputs
        if video_inputs:
            mm_data["video"] = video_inputs
        return {"prompt": prompt_text, "multi_modal_data": mm_data}

    def _sampling_params(self) -> Any:
        from vllm import SamplingParams

        return SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_new_tokens,
            n=1,
        )

    def _result(self, output: Any) -> LedgerUpdate:
        generation = output.outputs[0]
        normalized = normalize_evidence_ledger(
            generation.text.strip(),
            confirmed_cap=self.confirmed_cap,
            conflict_cap=self.conflict_cap,
        )
        return LedgerUpdate(
            text=normalized,
            num_tokens=len(generation.token_ids or []),
        )

    def update(
        self,
        retained_traces: List[Dict[str, Any]],
        question: str,
        options: List[str],
        image_path: str,
        max_traces_for_context: int = 4,
        verbose: bool = True,
        previous_ledger: Optional[str] = None,
        max_previous_ledger_chars: int = 8192,
    ) -> Optional[LedgerUpdate]:
        if not retained_traces:
            return None

        messages = build_evidence_ledger_messages(
            question=question,
            options=options,
            image_path=image_path,
            retained_traces=retained_traces,
            max_traces_for_context=max_traces_for_context,
            previous_ledger=previous_ledger,
            max_previous_ledger_chars=max_previous_ledger_chars,
            confirmed_cap=self.confirmed_cap,
            conflict_cap=self.conflict_cap,
        )
        start = time.time()
        try:
            outputs = self.llm.generate(
                [self._prepare(messages)],
                [self._sampling_params()],
            )
            result = self._result(outputs[0])
        except Exception as exc:
            print(f"[EvidenceLedgerUpdater] Generation failed: {exc}")
            return None

        if verbose:
            elapsed = time.time() - start
            print(
                f"\n[EvidenceLedgerUpdater] Updated ledger in {elapsed:.2f}s "
                f"({result.num_tokens} tokens):"
            )
            print(result.text[:600] + ("..." if len(result.text) > 600 else ""))
        return result

    def update_batch(
        self,
        items: List[Optional[Dict[str, Any]]],
        max_traces_for_context: int = 4,
        verbose: bool = True,
        max_previous_ledger_chars: int = 8192,
    ) -> List[Optional[LedgerUpdate]]:
        valid_indices: List[int] = []
        prompts: List[Dict[str, Any]] = []
        for index, item in enumerate(items):
            if item is None or not item.get("retained_traces"):
                continue
            messages = build_evidence_ledger_messages(
                question=item["question"],
                options=item["options"],
                image_path=item["image_path"],
                retained_traces=item["retained_traces"],
                max_traces_for_context=max_traces_for_context,
                previous_ledger=item.get("previous_ledger"),
                max_previous_ledger_chars=max_previous_ledger_chars,
                confirmed_cap=self.confirmed_cap,
                conflict_cap=self.conflict_cap,
            )
            try:
                prompts.append(self._prepare(messages))
                valid_indices.append(index)
            except Exception as exc:
                print(f"[EvidenceLedgerUpdater] Failed to build item {index}: {exc}")

        results: List[Optional[LedgerUpdate]] = [None] * len(items)
        if not prompts:
            return results

        start = time.time()
        try:
            outputs = self.llm.generate(
                prompts,
                [self._sampling_params()] * len(prompts),
            )
        except Exception as exc:
            print(f"[EvidenceLedgerUpdater] Batch generation failed: {exc}")
            return results

        for index, output in zip(valid_indices, outputs):
            try:
                results[index] = self._result(output)
            except Exception as exc:
                print(
                    f"[EvidenceLedgerUpdater] Failed to parse item {index}: {exc}"
                )
        if verbose:
            token_count = sum(result.num_tokens for result in results if result)
            print(
                f"  [EvidenceLedgerUpdater] Updated {len(outputs)} ledgers in "
                f"{time.time() - start:.2f}s ({token_count} tokens)"
            )
        return results
