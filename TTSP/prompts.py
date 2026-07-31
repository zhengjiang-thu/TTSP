"""Prompt builders for fresh exploration, guided exploration, and ledger updates."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


_TOOL_DEF = [
    {
        "type": "function",
        "function": {
            "name": "image_zoom_in_tool",
            "description": (
                "Crop and re-encode a region of an image so fine-grained visual "
                "evidence can be inspected directly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox_2d": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": (
                            "Bounding box [x1, y1, x2, y2] in [0, 1000] "
                            "normalized coordinates."
                        ),
                    },
                    "label": {
                        "type": "string",
                        "description": "Short name for the region or evidence sought.",
                    },
                    "img_idx": {
                        "type": "number",
                        "description": "Image index (0 is the original image).",
                    },
                },
                "required": ["bbox_2d", "label", "img_idx"],
            },
        },
    }
]

_TOOL_PROMPT = f"""

# Tools

You may call one or more functions to inspect the image.

Function signatures are provided inside <tools></tools> XML tags:
<tools>
{json.dumps(_TOOL_DEF, ensure_ascii=False)}
</tools>

Return each call as JSON inside <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


def get_perception_system_prompt() -> str:
    """Prompt for an independent fresh perception trace."""
    return """You are a visual research assistant. Answer the image-grounded question by alternating between observation, targeted inspection, and review.

1. Observe what is legible in the current view and identify the missing visual evidence needed for the question.
2. Use the crop tool on the region most likely to contain that evidence.
3. Review the returned view before deciding whether another inspection is needed.

Do not assume that an early crop is correct. Revisit the original image and inspect a different region when the acquired evidence does not resolve the question.

Finish with only the answer inside \\boxed{}. When options are supplied, select one of them and output only its letter, for example \\boxed{A}. For a free-form question, put a concise answer inside the box.""" + _TOOL_PROMPT


def get_perception_system_prompt_with_ledger(evidence_ledger: str) -> str:
    """Prompt for a guided trace conditioned on the Evidence Ledger."""
    return f"""You are a visual research assistant. Answer the image-grounded question by using the Evidence Ledger from earlier rounds and acquiring targeted new evidence.

## EVIDENCE LEDGER
{evidence_ledger}

How to use the ledger:
- CONFIRMED KNOWLEDGE is verified visual evidence. Reuse it without spending crops to re-check it unless the current image view directly contradicts it.
- OPEN CONFLICTS are prioritized inspection targets. Follow each spatial directive, inspect the named region, and determine which competing claim is supported.

Plan around the conflicts that could change the answer, inspect those regions with the crop tool, and combine the new observations with confirmed knowledge. If direct evidence contradicts a confirmed entry, state the contradiction so the next ledger update can demote it.

Finish with only the answer inside \\boxed{{}}. When options are supplied, select one of them and output only its letter, for example \\boxed{{A}}. For a free-form question, put a concise answer inside the box.""" + _TOOL_PROMPT


def get_perception_system_prompt_with_knowledge(knowledge_block: str) -> str:
    """Deprecated alias for the initial release's terminology."""
    return get_perception_system_prompt_with_ledger(knowledge_block)


def build_perception_messages(
    question: str,
    options: List[str],
    image_path: str,
    evidence_ledger: Optional[str] = None,
    fresh_exploration: bool = False,
    **legacy_kwargs: Any,
) -> List[Dict[str, Any]]:
    """Build messages for either the fresh or ledger-guided trace pool."""
    if evidence_ledger is None:
        evidence_ledger = legacy_kwargs.pop("visual_facts", None)
    if legacy_kwargs:
        unknown = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"unexpected prompt arguments: {unknown}")

    if evidence_ledger and not fresh_exploration:
        system_prompt = get_perception_system_prompt_with_ledger(evidence_ledger)
    else:
        system_prompt = get_perception_system_prompt()

    question_text = f"Question: {question}\n"
    if options:
        question_text += "Options:\n" + "".join(
            f"{chr(65 + index)}. {option}\n"
            for index, option in enumerate(options)
        )
    return [
        {"role": "system", "content": [{"text": system_prompt}]},
        {
            "role": "user",
            "content": [{"image": image_path}, {"text": question_text}],
        },
    ]


_EVIDENCE_LEDGER_SYSTEM = """You are the deterministic auditor that updates a two-tier Evidence Ledger for iterative visual reasoning.

Apply exactly these four rules:

1. RETAIN: carry forward every uncontradicted confirmed entry.
2. DEMOTE: when current visual evidence contradicts a confirmed entry, move it to OPEN CONFLICTS together with the competing claim and a spatial directive that can adjudicate them.
3. PROMOTE: when independent retained traces resolve an open conflict with sufficient agreement and direct image support, move the resolved claim to CONFIRMED KNOWLEDGE.
4. CONFIRM: admit a new claim only when it is consistent across independent retained traces, directly supported by the image, compatible with existing confirmed knowledge, and relevant to the question. Otherwise place a consequential disagreement in OPEN CONFLICTS with a concrete spatial directive.

Precision is more important than coverage. A missing fact costs another inspection round; a false confirmation biases every later guided trace.

Requirements:
- Every confirmed entry must contain a concise claim and the image region that supports it.
- Every open conflict must contain the competing claims and an actionable region to inspect next.
- Do not output an answer, option choice, answer-oriented reasoning, subjective interpretation, or irrelevant scene description.
- Do not refer to trace numbers in the final ledger.

Output exactly these headings:

## CONFIRMED KNOWLEDGE
1. Claim: <claim> | Evidence region: <region>

## OPEN CONFLICTS
1. Claims: <claim A> vs <claim B> | Inspect: <region and visual cue>
"""


def _trace_block(trace: Dict[str, Any], index: int, max_text_chars: int) -> str:
    texts = trace.get("texts", [trace.get("text", "")])
    steps = [
        f"[Step {step_index}]\n{text}"
        for step_index, text in enumerate(texts, start=1)
        if text
    ]
    content = "\n\n".join(steps) or trace.get("text", "") or "(empty trace)"
    if len(content) > max_text_chars:
        content = content[:max_text_chars] + "...[truncated]"
    score = trace.get("reliability_score")
    score_text = f" | reliability={score:.6f}" if isinstance(score, (int, float)) else ""
    return f"--- Retained Trace {index}{score_text} ---\n{content}"


def build_evidence_ledger_messages(
    question: str,
    options: List[str],
    image_path: str,
    retained_traces: List[Dict[str, Any]],
    previous_ledger: Optional[str] = None,
    max_traces_for_context: int = 4,
    max_text_chars: int = 4096,
    max_previous_ledger_chars: int = 8192,
    confirmed_cap: int = 8,
    conflict_cap: int = 4,
) -> List[Dict[str, Any]]:
    """Build the image-conditioned prompt for one greedy ledger update."""
    if max_traces_for_context < 1:
        raise ValueError("max_traces_for_context must be positive")
    selected = sorted(
        retained_traces,
        key=lambda trace: trace.get("reliability_score", float("-inf")),
        reverse=True,
    )[:max_traces_for_context]
    traces_block = "\n\n".join(
        _trace_block(trace, index, max_text_chars)
        for index, trace in enumerate(selected, start=1)
    )

    question_text = f"Question: {question}\n"
    if options:
        question_text += "Options:\n" + "".join(
            f"{chr(65 + index)}. {option}\n"
            for index, option in enumerate(options)
        )

    previous_section = ""
    if previous_ledger:
        previous = previous_ledger[:max_previous_ledger_chars]
        if len(previous_ledger) > len(previous):
            previous += "\n...[truncated]"
        previous_section = f"## PREVIOUS EVIDENCE LEDGER\n{previous}\n\n"

    user_text = (
        f"{question_text}\n"
        f"{previous_section}"
        f"## CURRENT RETAINED TRACES\n{traces_block}\n\n"
        "Update the ledger using RETAIN, DEMOTE, PROMOTE, and CONFIRM. "
        f"Return at most {confirmed_cap} confirmed entries and {conflict_cap} "
        "open conflicts. Use 'None identified.' when a tier is empty."
    )
    return [
        {"role": "system", "content": [{"text": _EVIDENCE_LEDGER_SYSTEM}]},
        {
            "role": "user",
            "content": [{"image": image_path}, {"text": user_text}],
        },
    ]


def build_knowledge_extraction_messages(
    question: str,
    options: List[str],
    image_path: str,
    filtered_traces: List[Dict[str, Any]],
    previous_knowledge: Optional[str] = None,
    max_traces_for_context: int = 4,
    max_text_chars: int = 4096,
    max_previous_knowledge_chars: int = 8192,
) -> List[Dict[str, Any]]:
    """Deprecated wrapper using the initial release's parameter names."""
    return build_evidence_ledger_messages(
        question=question,
        options=options,
        image_path=image_path,
        retained_traces=filtered_traces,
        previous_ledger=previous_knowledge,
        max_traces_for_context=max_traces_for_context,
        max_text_chars=max_text_chars,
        max_previous_ledger_chars=max_previous_knowledge_chars,
    )
