"""
TTSP Pipeline: Test-Time Scaling over Perception

Implements the full 4-stage iterative framework:
  1. Parallel Perception  - generate K independent traces with image zoom tool
  2. Reliability Filtering - discard traces with high token-level entropy
  3. Knowledge Extraction  - LLM self-reflection extracts confirmed visual facts
  4. Iterative Thinking    - feed facts back as prior context for next round

Rounds 1..N-1 run the full cycle; the final round skips extraction and goes
straight to voting over all accumulated filtered traces.
"""

import os
import time
import copy
import random
from functools import partial
from typing import Optional, List, Dict, Any

import numpy as np
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

from .inference import BatchInferenceLoop, TraceState, get_image_path
from .reliability import filter_traces, find_optimal_k
from .knowledge_extractor import KnowledgeExtractor
from .voting import compute_voting_results
from .tools import execute_tool_call, parse_tool_calls_from_response, cleanup_zoom_image
from .prompts import build_perception_messages
from .outputs import TTSPOutput, PerRoundResult
from .utils import process_trace_state


def _log_round_voting(round_idx: int, voting_results: Dict[str, Any], prefix: str = ""):
    """Print a compact one-line summary of cumulative voting after a round."""
    res = voting_results.get("TTSP")
    if res and res.get("answer"):
        print(f"  {prefix}Cumulative vote (round 0..{round_idx}): "
              f"TTSP={res['answer']}({res.get('num_votes', 0)})")


class _SampleContext:
    """Internal per-sample state tracked across TTSP rounds during batch processing."""

    def __init__(self, sample_id, question, options, base_image_path, sample_name=None):
        self.sample_id = sample_id
        self.question = question
        self.options = options
        self.base_image_path = base_image_path
        self.sample_name = sample_name
        self.visual_facts = None
        self.all_traces = []
        self.all_filtered_traces = []
        self.per_round_results = []
        self.generation_time = 0.0
        self.extraction_time = 0.0


class TTSPPipeline:
    """
    Main TTSP pipeline.

    Usage::

        pipeline = TTSPPipeline(model="/path/to/Qwen3-VL-8B-Thinking")
        output = pipeline.run(
            question="...", options=["...", ...], image_path="...",
            rounds=2, budget_per_round=8, sampling_params=SamplingParams(...)
        )
    """

    def __init__(self, model: str, **vllm_kwargs):
        self.model_name = model
        self.model_short_name = os.path.basename(model.rstrip('/'))
        self.run_id = f"{int(time.time() * 1e6)}_{random.randint(1, 9)}"

        default_kwargs = {
            "tensor_parallel_size": 4,
            "enable_prefix_caching": True,
            "trust_remote_code": True,
            "disable_log_stats": True,
        }
        if any(k in model.lower() for k in ('fp8', 'int8')):
            default_kwargs.update({"quantization": "fp8", "kv_cache_dtype": "fp8"})
        default_kwargs.update(vllm_kwargs)

        print("Initializing vLLM...")
        t0 = time.time()
        self.llm = LLM(model=model, **default_kwargs)
        print(f"vLLM ready in {time.time() - t0:.2f}s")

        print("Initializing processor...")
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        print(f"Processor ready in {time.time() - t0:.2f}s")

        self.knowledge_extractor = KnowledgeExtractor(self.llm, self.processor)
        self._tool_executor = partial(
            execute_tool_call, model_short_name=self.model_short_name
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(
        self,
        question: str,
        options: List[str],
        image_path: str,
        rounds: int = 2,
        budget_per_round: int = 8,
        batch_size: int = 8,
        sampling_params: Optional[SamplingParams] = None,
        filtering_ratio: float = 0.4,
        gamma: float = 0.1,
        reasoning_effort: str = "low",
        dataset_name: Optional[str] = None,
        sample_name: Optional[str] = None,
        save_zoom_images: bool = False,
        max_extraction_traces: int = 6,
        fresh_exploration_ratio: float = 0.4,
        max_previous_knowledge_chars: int = 8192,
    ) -> TTSPOutput:
        """
        Run the full TTSP iterative pipeline.

        Args:
            question: VQA question text.
            options: Answer choices (empty list for free-form).
            image_path: Path to the input image.
            rounds: Total number of iterative rounds (>=1).
            budget_per_round: Number of parallel perception traces per round.
            batch_size: vLLM batch size for trace generation.
            sampling_params: vLLM SamplingParams; defaults applied if None.
            filtering_ratio: Fraction of low-reliability traces to discard per round.
            gamma: Temperature for reliability-weighted voting.
            reasoning_effort: Passed to Qwen chat template ('low' / 'high').
            dataset_name: For tool image saving paths.
            sample_name: For tool image saving paths.
            save_zoom_images: Whether to keep zoom-in image files permanently.
            max_extraction_traces: Max traces fed into knowledge extractor.
            fresh_exploration_ratio: Fraction of traces to generate fresh (without knowledge) in each round.

        Returns:
            TTSPOutput with voting results and per-round details.
        """
        total_start = time.time()
        output = TTSPOutput(config={
            "model": self.model_name,
            "rounds": rounds,
            "budget_per_round": budget_per_round,
            "filtering_ratio": filtering_ratio,
            "gamma": gamma,
        })

        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.6, top_p=0.95, top_k=20, max_tokens=4096, logprobs=10
            )

        pil_image = Image.open(image_path)
        base_image_path = [get_image_path(pil_image)]

        visual_facts: Optional[str] = None  # updated after each round except last

        for round_idx in range(rounds):
            is_last_round = (round_idx == rounds - 1)
            print(f"\n{'='*70}")
            print(f"TTSP Round {round_idx + 1}/{rounds}"
                  + (" (final)" if is_last_round else " (with knowledge extraction)"))
            print('='*70)

            # ----- Stage 1: Parallel Perception -----
            round_result = PerRoundResult(round_idx=round_idx)
            gen_start = time.time()

            # Build messages with and without knowledge
            messages_with_knowledge = build_perception_messages(
                question, options, base_image_path[0], visual_facts, fresh_exploration=False
            )
            messages_fresh = build_perception_messages(
                question, options, base_image_path[0], visual_facts, fresh_exploration=True
            )

            # Calculate number of fresh traces for this round
            fresh_count = int(budget_per_round * fresh_exploration_ratio)

            traces = self._generate_traces(
                messages=messages_with_knowledge,
                base_image_path=base_image_path,
                budget=budget_per_round,
                batch_size=batch_size,
                sampling_params=sampling_params,
                reasoning_effort=reasoning_effort,
                dataset_name=dataset_name,
                sample_name=sample_name,
                save_zoom_images=save_zoom_images,
                round_offset=round_idx * budget_per_round,
                fresh_messages=messages_fresh,
                fresh_count=fresh_count,
            )

            round_result.generation_time = time.time() - gen_start
            round_result.traces = traces
            round_result.total_tokens = sum(t.get('num_tokens', 0) for t in traces)

            # Tag each trace with round info so downstream can distinguish rounds
            for t in traces:
                t['round_idx'] = round_idx
                t['visual_facts_input'] = visual_facts  # facts used as context for this round

            output.generation_time += round_result.generation_time
            output.all_traces.extend(traces)

            # ----- Stage 2: Reliability Filtering -----
            filtered, threshold = filter_traces(traces, filtering_ratio=filtering_ratio)
            n_filtered_out = len(traces) - len(filtered)
            round_result.filtered_traces = filtered
            round_result.filtering_ratio = n_filtered_out / max(len(traces), 1)
            output.all_filtered_traces.extend(filtered)

            n_with_ans = sum(1 for t in filtered if t.get('extracted_answer'))
            print(f"  Filtering: {len(traces)} → {len(filtered)} traces "
                  f"({n_filtered_out} removed, threshold={threshold:.4f})")
            print(f"  Traces with answer: {n_with_ans}/{len(filtered)}")

            # ----- Stage 3: Knowledge Extraction (skip on last round) -----
            if not is_last_round and filtered:
                ext_start = time.time()
                print("  Running knowledge extraction...")
                visual_facts = self.knowledge_extractor.extract(
                    filtered_traces=filtered,
                    question=question,
                    options=options,
                    image_path=base_image_path[0],
                    max_traces_for_context=max_extraction_traces,
                    previous_knowledge=visual_facts,
                    max_previous_knowledge_chars=max_previous_knowledge_chars,
                )
                output.extraction_time += time.time() - ext_start
                round_result.visual_facts = visual_facts

            # ----- Per-round cumulative voting (to track knowledge effect) -----
            round_result.voting_results = compute_voting_results(
                output.all_traces,
                filtering_ratio=filtering_ratio,
                gamma=gamma,
            )
            _log_round_voting(round_idx, round_result.voting_results)

            output.per_round_results.append(round_result)

        # ----- Stage 4: Final Voting over all filtered traces -----
        output.voting_results = compute_voting_results(
            output.all_traces,
            filtering_ratio=filtering_ratio,
            gamma=gamma,
        )
        output.total_traces_count = len(output.all_traces)
        output.total_tokens = sum(t.get('num_tokens', 0) for t in output.all_traces)
        output.avg_tokens_per_trace = (
            output.total_tokens / output.total_traces_count
            if output.total_traces_count else 0.0
        )
        output.total_time = time.time() - total_start
        output.print_summary()
        return output

    def run_batch(
        self,
        samples: List[Dict[str, Any]],
        rounds: int = 2,
        budget_per_round: int = 8,
        batch_size: int = 64,
        sampling_params: Optional[SamplingParams] = None,
        filtering_ratio: float = 0.4,
        gamma: float = 0.1,
        reasoning_effort: str = "low",
        dataset_name: Optional[str] = None,
        save_zoom_images: bool = False,
        max_extraction_traces: int = 6,
        save_debug_details: bool = False,
        fresh_exploration_ratio: float = 0.4,
        max_previous_knowledge_chars: int = 8192,
    ) -> List[TTSPOutput]:
        """
        Run the full TTSP pipeline on multiple samples with cross-sample batching.

        All samples share the same round schedule: in each round, traces for
        every sample are generated in one large LLM batch, then filtered and
        extracted per-sample. This dramatically improves GPU utilisation compared
        to processing one sample at a time.

        Args:
            samples: List of dicts, each must have keys:
                     'question' (str), 'options' (list),
                     'image_path' or 'img_path' (str).
                     Optional: 'sample_name' (str).
            rounds: Total iterative rounds (>=1).
            budget_per_round: Parallel perception traces per sample per round.
            batch_size: Max traces per LLM generate call.
            sampling_params: vLLM SamplingParams; defaults applied if None.
            filtering_ratio: Fraction of low-reliability traces to discard.
            gamma: Temperature for reliability-weighted voting.
            reasoning_effort: Passed to Qwen chat template.
            dataset_name: For tool image saving paths.
            save_zoom_images: Keep zoom-in images permanently.
            max_extraction_traces: Max traces fed into knowledge extractor.
            fresh_exploration_ratio: Fraction of traces to generate fresh (without knowledge) in each round.

        Returns:
            List of TTSPOutput, one per input sample.
        """
        total_start = time.time()
        n_samples = len(samples)

        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.6, top_p=0.95, top_k=20, max_tokens=4096, logprobs=10,
            )

        # ---- Initialise per-sample contexts ----
        contexts: List[_SampleContext] = []
        for i, sample in enumerate(samples):
            img_path = sample.get('image_path') or sample['img_path']
            pil_image = Image.open(img_path)
            base_path = [get_image_path(pil_image)]
            ctx = _SampleContext(
                sample_id=i,
                question=sample['question'],
                options=sample.get('options', []),
                base_image_path=base_path,
                sample_name=sample.get('sample_name', f'sample_{i}'),
            )
            contexts.append(ctx)

        inference_loop = BatchInferenceLoop(
            llm=self.llm,
            processor=self.processor,
            tool_executor_func=self._tool_executor,
            tool_parser_func=parse_tool_calls_from_response,
        )

        base_seed = time.time_ns()

        # ---- Round loop (synchronised across all samples) ----
        for round_idx in range(rounds):
            is_last_round = (round_idx == rounds - 1)
            print(f"\n{'='*70}")
            print(f"TTSP Batch Round {round_idx + 1}/{rounds} ({n_samples} samples)"
                  + (" (final)" if is_last_round else " (with knowledge extraction)"))
            print('=' * 70)

            # --- Stage 1: build trace states for ALL samples ---
            gen_start = time.time()
            all_states: List[TraceState] = []
            # (start_idx, end_idx) in all_states for each sample
            sample_ranges: List[tuple] = []

            # Calculate number of fresh traces per sample
            fresh_count = int(budget_per_round * fresh_exploration_ratio)

            for ctx in contexts:
                # Build messages with and without knowledge
                messages_with_knowledge = build_perception_messages(
                    ctx.question, ctx.options,
                    ctx.base_image_path[0], ctx.visual_facts,
                    fresh_exploration=False,
                )
                messages_fresh = build_perception_messages(
                    ctx.question, ctx.options,
                    ctx.base_image_path[0], ctx.visual_facts,
                    fresh_exploration=True,
                )
                start_idx = len(all_states)
                for trace_i in range(budget_per_round):
                    global_idx = (
                        ctx.sample_id * rounds * budget_per_round
                        + round_idx * budget_per_round
                        + trace_i
                    )
                    params = copy.deepcopy(sampling_params)
                    params.n = 1
                    params.logprobs = 10
                    params.seed = base_seed + global_idx

                    # First 'fresh_count' traces for each sample are fresh
                    is_fresh = trace_i < fresh_count
                    use_messages = messages_fresh if is_fresh else messages_with_knowledge

                    state = TraceState(
                        trace_id=global_idx,
                        initial_messages=use_messages,
                        sampling_params=params,
                        image_paths=ctx.base_image_path,
                        sample_name=ctx.sample_name,
                        is_fresh=is_fresh,
                    )
                    all_states.append(state)
                sample_ranges.append((start_idx, len(all_states)))

            total_traces = len(all_states)
            print(f"  Generating {total_traces} traces "
                  f"({n_samples} samples × {budget_per_round} traces) ...")

            # --- Run multi-turn inference in batches ---
            for b_start in range(0, total_traces, batch_size):
                batch = all_states[b_start: b_start + batch_size]
                inference_loop.run(
                    trace_states=batch,
                    max_turns=10,
                    reasoning_effort=reasoning_effort,
                    dataset_name=dataset_name,
                    sample_name=None,  # per-trace sample_name used instead
                    run_id=self.run_id,
                    batch_start_idx=b_start,
                    save_zoom_images=save_zoom_images,
                )

            gen_time = time.time() - gen_start
            print(f"  Generation done in {gen_time:.2f}s")

            # --- Stage 2: per-sample processing & filtering ---
            for ctx_i, ctx in enumerate(contexts):
                s_start, s_end = sample_ranges[ctx_i]
                sample_states = all_states[s_start:s_end]

                # Clean up zoom images
                if not save_zoom_images:
                    for state in sample_states:
                        for path in state.image_paths[len(ctx.base_image_path):]:
                            cleanup_zoom_image(path)

                traces = [process_trace_state(s) for s in sample_states]

                # Tag each trace with round info
                for t in traces:
                    t['round_idx'] = round_idx
                    t['visual_facts_input'] = ctx.visual_facts

                round_result = PerRoundResult(round_idx=round_idx)
                round_result.traces = traces
                round_result.total_tokens = sum(t.get('num_tokens', 0) for t in traces)
                round_result.generation_time = gen_time / n_samples

                ctx.all_traces.extend(traces)
                ctx.generation_time += round_result.generation_time

                # Reliability filtering
                filtered, threshold = filter_traces(traces, filtering_ratio=filtering_ratio)
                n_removed = len(traces) - len(filtered)
                round_result.filtered_traces = filtered
                round_result.filtering_ratio = n_removed / max(len(traces), 1)
                ctx.all_filtered_traces.extend(filtered)

                n_with_ans = sum(1 for t in filtered if t.get('extracted_answer'))
                print(f"  [{ctx.sample_name}] {len(traces)}→{len(filtered)} traces "
                      f"({n_removed} removed), {n_with_ans} with answers")

                # Per-round cumulative voting (to track knowledge effect)
                round_result.voting_results = compute_voting_results(
                    ctx.all_traces,
                    filtering_ratio=filtering_ratio,
                    gamma=gamma,
                )

                ctx.per_round_results.append(round_result)

            # --- Stage 3: batch knowledge extraction (skip last round) ---
            if not is_last_round:
                ext_start = time.time()
                print(f"  Batch knowledge extraction for {n_samples} samples ...")

                extraction_items: List[Optional[Dict]] = []
                for ctx in contexts:
                    latest_filtered = ctx.per_round_results[-1].filtered_traces
                    if latest_filtered:
                        extraction_items.append({
                            'filtered_traces': latest_filtered,
                            'question': ctx.question,
                            'options': ctx.options,
                            'image_path': ctx.base_image_path[0],
                            'previous_knowledge': ctx.visual_facts,
                        })
                    else:
                        extraction_items.append(None)

                all_facts = self.knowledge_extractor.extract_batch(
                    extraction_items,
                    max_traces_for_context=max_extraction_traces,
                    max_previous_knowledge_chars=max_previous_knowledge_chars,
                )

                ext_time = time.time() - ext_start
                for ctx, facts in zip(contexts, all_facts):
                    ctx.visual_facts = facts
                    ctx.per_round_results[-1].visual_facts = facts
                    ctx.extraction_time += ext_time / n_samples

        # ---- Stage 4: per-sample voting & output assembly ----
        outputs: List[TTSPOutput] = []
        for ctx in contexts:
            output = TTSPOutput(config={
                "model": self.model_name,
                "rounds": rounds,
                "budget_per_round": budget_per_round,
                "filtering_ratio": filtering_ratio,
                "gamma": gamma,
                "batch_mode": True,
            })
            output.per_round_results = ctx.per_round_results
            output.all_traces = ctx.all_traces
            output.all_filtered_traces = ctx.all_filtered_traces
            output.generation_time = ctx.generation_time
            output.extraction_time = ctx.extraction_time

            output.voting_results = compute_voting_results(
                ctx.all_traces,
                filtering_ratio=filtering_ratio,
                gamma=gamma,
            )
            output.total_traces_count = len(ctx.all_traces)
            output.total_tokens = sum(t.get('num_tokens', 0) for t in ctx.all_traces)
            output.avg_tokens_per_trace = (
                output.total_tokens / output.total_traces_count
                if output.total_traces_count else 0.0
            )
            output.total_time = time.time() - total_start
            outputs.append(output)

        total_time = time.time() - total_start
        total_traces = sum(o.total_traces_count for o in outputs)
        total_tokens = sum(o.total_tokens for o in outputs)
        print(f"\n{'='*70}")
        print(f"TTSP Batch Complete: {n_samples} samples, {rounds} rounds")
        print(f"Total traces: {total_traces}, Total tokens: {total_tokens:,}")
        print(f"Total time: {total_time:.2f}s "
              f"({total_time / n_samples:.2f}s/sample)")
        print('=' * 70)

        return outputs

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _generate_traces(
        self,
        messages: List[Dict],
        base_image_path: List[str],
        budget: int,
        batch_size: int,
        sampling_params: SamplingParams,
        reasoning_effort: str,
        dataset_name: Optional[str],
        sample_name: Optional[str],
        save_zoom_images: bool,
        round_offset: int = 0,
        fresh_messages: Optional[List[Dict]] = None,
        fresh_count: int = 0,
    ) -> List[Dict[str, Any]]:
        """Generate `budget` parallel traces in batches and return processed trace dicts.

        Args:
            messages: List of message dicts for traces with knowledge.
            fresh_messages: List of message dicts for traces without knowledge (fresh exploration).
                           If None, all traces use `messages`.
            fresh_count: Number of traces to generate without knowledge (fresh exploration).
                        These will be generated first.
        """
        inference_loop = BatchInferenceLoop(
            llm=self.llm,
            processor=self.processor,
            tool_executor_func=self._tool_executor,
            tool_parser_func=parse_tool_calls_from_response,
        )

        base_seed = time.time_ns()
        all_states = []

        # Determine which traces should be fresh (without knowledge)
        # First 'fresh_count' traces will be fresh
        fresh_indices = set(range(fresh_count))

        for batch_start in range(0, budget, batch_size):
            batch_end = min(batch_start + batch_size, budget)
            batch_states = []

            for offset in range(batch_end - batch_start):
                global_idx = round_offset + batch_start + offset
                params = copy.deepcopy(sampling_params)
                params.n = 1
                params.logprobs = 10
                params.seed = base_seed + global_idx

                # Use fresh messages (no knowledge) for fresh traces
                is_fresh = global_idx in fresh_indices
                use_messages = fresh_messages if is_fresh and fresh_messages else messages
                # Tag the state so we can identify fresh traces later
                state = TraceState(
                    trace_id=global_idx,
                    initial_messages=use_messages,
                    sampling_params=params,
                    image_paths=base_image_path,
                    is_fresh=is_fresh,
                )
                batch_states.append(state)

            inference_loop.run(
                trace_states=batch_states,
                max_turns=10,
                reasoning_effort=reasoning_effort,
                dataset_name=dataset_name,
                sample_name=sample_name,
                run_id=self.run_id,
                batch_start_idx=batch_start,
                save_zoom_images=save_zoom_images,
            )

            all_states.extend(batch_states)

            # Clean up zoom images unless saving permanently
            if not save_zoom_images:
                for state in batch_states:
                    for path in state.image_paths[len(base_image_path):]:
                        cleanup_zoom_image(path)

        traces = [process_trace_state(s) for s in all_states]
        n_answered = sum(1 for t in traces if t.get('extracted_answer'))
        n_fresh = sum(1 for t in traces if t.get('is_fresh', False))
        print(f"  Generated {len(traces)} traces ({n_fresh} fresh), {n_answered} with answers")
        return traces
