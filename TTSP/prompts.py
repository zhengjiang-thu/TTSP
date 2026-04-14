"""
Prompts for TTSP:
  - Parallel perception (visual exploration with tool use)
  - Knowledge extraction (LLM critical review to extract consensus + conflicts)
  - Iterative context building (prepend knowledge to next round)
"""
import json
from typing import List, Dict, Optional


# ============= TOOL DEFINITION =============

_TOOL_DEF = [
    {
        "type": "function",
        "function": {
            "name": "image_zoom_in_tool",
            "description": (
                "Zoom in on a specific region of an image by cropping it based on a "
                "bounding box. Use this to inspect fine-grained visual details."
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
                            "Bounding box [x1, y1, x2, y2] in [0, 1000] normalized coordinates, "
                            "where (x1,y1) is top-left and (x2,y2) is bottom-right."
                        ),
                    },
                    "label": {
                        "type": "string",
                        "description": "Name or label of the object/region to zoom into.",
                    },
                    "img_idx": {
                        "type": "number",
                        "description": "Index of the image to zoom into (0 = original image).",
                    },
                },
                "required": ["bbox_2d", "label", "img_idx"],
            },
        },
    }
]

_TOOL_PROMPT = f"""

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{json.dumps(_TOOL_DEF, ensure_ascii=False)}
</tools>

For each function call, return a JSON object within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


# ============= PERCEPTION PROMPTS =============

def get_perception_system_prompt() -> str:
    """System prompt for the parallel perception stage (Round 1)."""
    return """Your role is that of a research assistant specializing in visual information. Answer questions about images by looking at them closely and providing detailed analysis. Please follow this structured thinking process and show your work.

    Start an iterative loop for each question:

    - **First, look closely:** Begin with a detailed description of the image, paying attention to the user's question. List what you can tell just by looking, and what you'll need to look up.
    - **Next, find information:** Use a tool to research the things you need to find out.
    - **Then, review the findings:** Carefully analyze what the tool tells you and decide on your next action.

    Continue this loop until your research is complete.

    To finish, put your final answer within \\boxed{}, and make sure it contains only the answer itself without extra words or symbols.

    For multiple-choice questions, answer with the option's letter from the given choices directly, e.g. \\boxed{A}, \\boxed{B}, \\boxed{C}, \\boxed{D} etc. Note that YOU MUST choose One Answer from the Options.""" + _TOOL_PROMPT


def get_perception_system_prompt_with_knowledge(knowledge_block: str) -> str:
    """
    System prompt for later rounds.

    Two-tier structure:
      - CONFIRMED KNOWLEDGE: unanimously agreed and critically verified → use as foundation
      - OPEN CONFLICTS: disagreements between traces → the primary investigation targets
    """
    return f"""Your role is that of a research assistant specializing in visual information. Answer questions about images by looking at them closely and providing detailed analysis. Please follow this structured thinking process and show your work.

    Prior rounds of visual exploration have produced the following structured knowledge:

    {knowledge_block}

    How to use this:
    - **CONFIRMED KNOWLEDGE** items have been agreed upon by all prior exploration paths and critically reviewed. Treat them as reliable foundations — build on them directly without re-checking.
    - **OPEN CONFLICTS** are specific points where prior exploration paths disagreed. These are your PRIMARY investigation targets this round. For each conflict, zoom into the relevant region, examine the evidence carefully, and determine which side is correct (or whether both are wrong).

    Your goal this round is to RESOLVE the open conflicts and discover any NEW details that prior rounds did not cover.

    Start an iterative loop:

    - **First, plan:** Read the open conflicts. Decide which ones are most critical to answering the question, and plan which image regions to inspect.
    - **Next, investigate:** Use the zoom tool to directly examine the disputed regions. Gather clear visual evidence to settle each conflict.
    - **Then, synthesize:** Combine the confirmed knowledge with your new findings on the conflicts. If you happen to notice something that contradicts a confirmed fact, note it — but this should be rare.

    Continue this loop until your research is complete.

    To finish, put your final answer within \\boxed{{}}, and make sure it contains only the answer itself without extra words or symbols.

    For multiple-choice questions, answer with the option's letter from the given choices directly, e.g. \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, \\boxed{{D}} etc. Note that YOU MUST choose One Answer from the Options.""" + _TOOL_PROMPT


def build_perception_messages(
    question: str,
    options: List[str],
    image_path: str,
    visual_facts: Optional[str] = None,
    fresh_exploration: bool = False,
) -> List[Dict]:
    """
    Build the initial message list for the perception stage.

    Args:
        question: The question text.
        options: List of answer options (may be empty).
        image_path: Path to the image.
        visual_facts: Extracted structured knowledge from prior rounds (if any).
        fresh_exploration: If True, ignore visual_facts and use the base prompt.
    """
    if visual_facts and not fresh_exploration:
        system_prompt = get_perception_system_prompt_with_knowledge(visual_facts)
    else:
        system_prompt = get_perception_system_prompt()

    question_text = f"Question: {question}\n"
    if options:
        option_str = "".join(f"{chr(65+i)}. {opt}\n" for i, opt in enumerate(options))
        question_text += f"Options:\n{option_str}"

    return [
        {"role": "system", "content": [{"text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"image": image_path},
                {"text": question_text},
            ],
        },
    ]


# ============= KNOWLEDGE EXTRACTION =============
_KNOWLEDGE_EXTRACTION_SYSTEM = """You are a critical visual knowledge auditor for an iterative test-time perception system.

You will receive:
- The CURRENT ROUND's independent exploration traces (multiple traces for the same image and question)
- Optionally, the PREVIOUS ROUND's structured knowledge output (confirmed knowledge + open conflicts)

Your job is to produce an UPDATED knowledge state by performing four operations:

(A) INHERIT previous confirmed knowledge.
    - Carry forward each previously confirmed fact UNLESS a current-round trace provides direct visual evidence that contradicts it.
    - If contradicted, move it to OPEN CONFLICTS with the new competing claim.
    - Do NOT drop a previously confirmed fact just because current traces don't mention it — they may have taken it for granted.

(B) RESOLVE previous open conflicts.
    - Check whether current-round traces now agree on a resolution for any previously open conflict.
    - If resolved: promote to CONFIRMED KNOWLEDGE or remove it.
    - If still unresolved or new evidence makes it more complex: keep in OPEN CONFLICTS with updated details.

(C) ADD new confirmed facts from the current round.
    - Apply the same four checks: Unanimity, Grounding, Plausibility, Relevance.
      1. Unanimity: Every current-round trace that mentions this topic agrees (or no trace contradicts it).
      2. Grounding: Directly visible in the image, not an inference.
      3. Plausibility: Internally consistent, not a likely hallucination.
      4. Relevance: Directly useful for answering the question.
    - All four must pass.

(D) FLAG new conflicts from the current round.
    - If current-round traces disagree on a NEW point relevant to the question, add it to OPEN CONFLICTS.

Rules:
- Be ruthlessly selective for CONFIRMED KNOWLEDGE. It is far better to confirm 2 solid facts than to let 1 wrong fact slip through. Wrong facts poison all subsequent rounds.
- Prioritize concrete, reusable observations that are directly relevant to resolving the question. Every fact you output should bring the system closer to the answer.
- Do NOT include ANY final answer, option selection, or reasoning toward an answer.
- Do NOT include subjective interpretations (e.g., "the scene looks peaceful") — only concrete visual observations.
- Do NOT include generic scene descriptions that do not help answer the question.
- If all traces agree but the claim seems implausible or suspiciously specific, OMIT it from confirmed knowledge and optionally flag it as needing verification under OPEN CONFLICTS.
- If traces simply discuss different aspects of the image without contradiction, those are NOT conflicts. Only report genuine disagreements relevant to the question.
- If there are no conflicts, the OPEN CONFLICTS section should say "None identified."
- Do NOT reference trace numbers — the next round cannot see the original traces.

Output format (follow exactly):

## CONFIRMED KNOWLEDGE
1. <fact>
2. ...

## OPEN CONFLICTS
1. <what the disagreement is about> | <competing claims> | <where to look in the image>
2. ...
"""


def build_knowledge_extraction_messages(
    question: str,
    options: List[str],
    image_path: str,
    filtered_traces: List[Dict],
    previous_knowledge: Optional[str] = None,
    max_traces_for_context: int = 6,
    max_text_chars: int = 4096,
    max_previous_knowledge_chars: int = 8192,
) -> List[Dict]:
    question_text = f"Question: {question}\n"
    if options:
        question_text += "Options:\n" + "".join(
            f"{chr(65+i)}. {opt}\n" for i, opt in enumerate(options)
        )

    traces_to_use = filtered_traces[:max_traces_for_context]
    trace_summaries = []
    total_traces = len(traces_to_use)

    for i, trace in enumerate(traces_to_use):
        texts = trace.get("texts", [trace.get("text", "")])

        step_texts = []
        for j, t in enumerate(texts):
            if not t:
                continue
            step_texts.append(f"[Step {j+1}]\n{t}")

        full_text = "\n\n".join(step_texts)

        if not full_text:
            fallback_text = trace.get("text", "")
            full_text = fallback_text if fallback_text else "(empty trace)"

        if len(full_text) > max_text_chars:
            full_text = full_text[:max_text_chars] + "...[truncated]"

        trace_summaries.append(f"--- Independent Trace {i+1} ---\n{full_text}")

    traces_block = "\n\n".join(trace_summaries) if trace_summaries else "(no traces available)"

    # Build the previous knowledge section (with length limit to prevent prompt overflow)
    if previous_knowledge:
        if len(previous_knowledge) > max_previous_knowledge_chars:
            previous_knowledge = (
                previous_knowledge[:max_previous_knowledge_chars]
                + "\n...[truncated due to length]"
            )
        prev_section = (
            "## PREVIOUS ROUND'S KNOWLEDGE OUTPUT\n"
            "The following was produced by the previous round's audit. Use it as the "
            "baseline to update — inherit, resolve, or revise as appropriate:\n\n"
            f"{previous_knowledge}\n\n"
            "---\n\n"
        )
    else:
        prev_section = ""

    user_text = (
        f"{question_text}\n"
        f"{prev_section}"
        f"## CURRENT ROUND TRACES\n"
        f"Below are {total_traces} independent perception traces from the current round.\n\n"
        f"{traces_block}\n\n"
        "Produce an updated knowledge state:\n\n"
        "## CONFIRMED KNOWLEDGE\n"
        "Inherit all previously confirmed facts unless contradicted by current traces. "
        "Add new facts that all current traces agree on and that pass plausibility and relevance checks. "
        "Promote any resolved conflicts. Be extremely selective — a wrong fact here will "
        "mislead all future rounds.\n\n"
        "## OPEN CONFLICTS\n"
        "Keep unresolved previous conflicts (with updated details if new evidence emerged). "
        "Add any new disagreements from current traces that matter for answering the question. "
        "For each, note what the disagreement is, the competing claims, and where to look in the image. "
        "Ignore disagreements about details irrelevant to the question.\n\n"
        "Do NOT provide the final answer or select an option. "
        "Do NOT include reasoning or interpretations — only visual observations."
    )

    return [
        {"role": "system", "content": [{"text": _KNOWLEDGE_EXTRACTION_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"image": image_path},
                {"text": user_text},
            ],
        },
    ]