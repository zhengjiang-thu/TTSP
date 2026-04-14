"""
Knowledge Extractor: Stage 3 of the TTSP pipeline.

Uses the same LLM (greedy, single pass) to reflect on filtered perception traces
and extract confirmed "Visual Facts" - precise, cross-validated observations about
the image that serve as prior context for subsequent reasoning rounds.
"""

import time
from typing import List, Dict, Any, Optional
from vllm import SamplingParams
from qwen_vl_utils import process_vision_info

from .prompts import build_knowledge_extraction_messages


class KnowledgeExtractor:
    """
    Runs a single greedy LLM inference to synthesize visual facts from
    multiple filtered perception traces.
    """

    def __init__(self, llm, processor, max_new_tokens: int = 1024):
        self.llm = llm
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    def extract(
        self,
        filtered_traces: List[Dict[str, Any]],
        question: str,
        options: List[str],
        image_path: str,
        max_traces_for_context: int = 6,
        verbose: bool = True,
        previous_knowledge: Optional[str] = None,
        max_previous_knowledge_chars: int = 8192,
    ) -> Optional[str]:
        """
        Synthesize visual facts from filtered traces.

        Args:
            filtered_traces: High-reliability perception traces from Stage 2.
            question: The original VQA question.
            options: Answer options (empty for free-form).
            image_path: Path to the original image.
            max_traces_for_context: How many traces to include in the extraction prompt.
            verbose: Print extracted facts to stdout.
            previous_knowledge: Structured knowledge from previous round (if any).

        Returns:
            A string of confirmed visual facts, or None if extraction fails.
        """
        if not filtered_traces:
            return None

        start = time.time()

        messages = build_knowledge_extraction_messages(
            question=question,
            options=options,
            image_path=image_path,
            filtered_traces=filtered_traces,
            max_traces_for_context=max_traces_for_context,
            previous_knowledge=previous_knowledge,
            max_previous_knowledge_chars=max_previous_knowledge_chars,
        )

        # Greedy decoding (no sampling): we want a deterministic, factual summary
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_new_tokens,
            n=1,
        )

        try:
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="low",  # no extended thinking for extraction
            )
            image_inputs, video_inputs = process_vision_info(messages)
            mm_data = {}
            if image_inputs:
                mm_data['image'] = image_inputs
            if video_inputs:
                mm_data['video'] = video_inputs

            outputs = self.llm.generate(
                [{'prompt': prompt_text, 'multi_modal_data': mm_data}],
                [sampling_params],
            )
            facts_text = outputs[0].outputs[0].text.strip()

        except Exception as e:
            print(f"[KnowledgeExtractor] Generation failed: {e}")
            return None

        elapsed = time.time() - start
        if verbose:
            print(f"\n[KnowledgeExtractor] Extracted facts in {elapsed:.2f}s:")
            print(facts_text[:600] + ("..." if len(facts_text) > 600 else ""))

        return facts_text

    def extract_batch(
        self,
        items: List[Optional[Dict[str, Any]]],
        max_traces_for_context: int = 6,
        verbose: bool = True,
        max_previous_knowledge_chars: int = 8192,
    ) -> List[Optional[str]]:
        """
        Batch extraction of visual facts for multiple samples in one LLM call.

        Args:
            items: List of dicts, each with keys:
                   'filtered_traces', 'question', 'options', 'image_path', 'previous_knowledge'.
                   None entries are skipped and return None.
            max_traces_for_context: Max traces per sample fed into the prompt.
            verbose: Print summary to stdout.

        Returns:
            List of extracted fact strings (or None) aligned with input items.
        """
        n_items = len(items)

        # Build prompts for all valid items
        valid_indices: List[int] = []
        all_prompts: List[Dict] = []

        for i, item in enumerate(items):
            if item is None or not item.get('filtered_traces'):
                continue

            messages = build_knowledge_extraction_messages(
                question=item['question'],
                options=item['options'],
                image_path=item['image_path'],
                filtered_traces=item['filtered_traces'],
                max_traces_for_context=max_traces_for_context,
                previous_knowledge=item.get('previous_knowledge'),
                max_previous_knowledge_chars=max_previous_knowledge_chars,
            )

            try:
                prompt_text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    reasoning_effort="low",
                )
                image_inputs, video_inputs = process_vision_info(messages)
                mm_data = {}
                if image_inputs:
                    mm_data['image'] = image_inputs
                if video_inputs:
                    mm_data['video'] = video_inputs

                all_prompts.append({'prompt': prompt_text, 'multi_modal_data': mm_data})
                valid_indices.append(i)
            except Exception as e:
                print(f"[KnowledgeExtractor] Failed to build prompt for item {i}: {e}")

        if not all_prompts:
            return [None] * n_items

        # Greedy decoding for all prompts in one batch
        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=self.max_new_tokens, n=1,
        )

        start = time.time()
        try:
            outputs = self.llm.generate(
                all_prompts,
                [sampling_params] * len(all_prompts),
            )
        except Exception as e:
            print(f"[KnowledgeExtractor] Batch generation failed: {e}")
            return [None] * n_items

        elapsed = time.time() - start
        if verbose:
            print(f"  [KnowledgeExtractor] Batch extracted {len(outputs)} facts in {elapsed:.2f}s")

        # Map results back to original positions
        results: List[Optional[str]] = [None] * n_items
        for idx, output in zip(valid_indices, outputs):
            results[idx] = output.outputs[0].text.strip()

        return results
