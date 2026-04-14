"""
Batch inference loop for TTSP parallel perception.
Handles multi-turn generation with image_zoom_in_tool calls.
"""
import copy
import json
import os
import tempfile
from typing import List, Optional, Callable, Dict, Any, Tuple
from vllm import SamplingParams
from qwen_vl_utils import process_vision_info
from PIL import Image


def get_image_path(pil_image: Image.Image) -> str:
    """Return file path for a PIL image, saving to a temp file if needed."""
    if hasattr(pil_image, 'filename') and pil_image.filename:
        return pil_image.filename
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        pil_image.save(tmp.name)
        return tmp.name


class TraceState:
    """State for a single perception trace during inference."""

    def __init__(
        self,
        trace_id: int,
        initial_messages: List[Dict],
        sampling_params: SamplingParams,
        image_paths: List[str],
        sample_name: Optional[str] = None,
        is_fresh: bool = False,
    ):
        self.trace_id = trace_id
        self.messages = copy.deepcopy(initial_messages)
        self.sampling_params = sampling_params
        self.image_paths = copy.deepcopy(image_paths)
        self.sample_name = sample_name  # per-trace sample name for cross-sample batching
        self.is_fresh = is_fresh  # whether this trace was generated without knowledge

        self.vllm_outputs: List[Any] = []
        self.turn_texts: List[str] = []
        self.tool_bboxes: List[Any] = []

        self.is_finished = False
        self.error = None


class BatchInferenceLoop:
    """
    Multi-turn inference loop for a batch of traces.
    On each turn, generates from all active traces in parallel, detects tool calls,
    executes them, and feeds results back into the conversation.
    """

    def __init__(
        self,
        llm,
        processor,
        tool_executor_func: Callable,
        tool_parser_func: Callable,
    ):
        self.llm = llm
        self.processor = processor
        self.tool_executor = tool_executor_func
        self.tool_parser = tool_parser_func

    def run(
        self,
        trace_states: List[TraceState],
        max_turns: int = 10,
        reasoning_effort: str = "low",
        dataset_name: Optional[str] = None,
        sample_name: Optional[str] = None,
        run_id: Optional[str] = None,
        batch_start_idx: int = 0,
        save_zoom_images: bool = False,
    ) -> List[TraceState]:
        for turn in range(1, max_turns + 1):
            active = [s for s in trace_states if not s.is_finished]
            if not active:
                break
            if turn > 1:
                print(f"  Turn {turn}: {len(active)} active traces continuing...")

            prompts, valid = self._prepare_inputs(active, reasoning_effort)
            if not prompts:
                break

            outputs = self.llm.generate(prompts, [s.sampling_params for s in valid])

            for state, vllm_out in zip(valid, outputs):
                self._process_output(
                    state, vllm_out,
                    dataset_name, sample_name, run_id, turn, save_zoom_images
                )
        return trace_states

    def _prepare_inputs(
        self, active_states: List[TraceState], reasoning_effort: str
    ) -> Tuple[List[Dict], List[TraceState]]:
        prompts, valid = [], []
        for state in active_states:
            if not state.messages:
                continue
            prompt_text = self.processor.apply_chat_template(
                state.messages,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort=reasoning_effort,
            )
            image_inputs, video_inputs = process_vision_info(state.messages)
            mm_data = {}
            if image_inputs:
                mm_data['image'] = image_inputs
            if video_inputs:
                mm_data['video'] = video_inputs
            prompts.append({'prompt': prompt_text, 'multi_modal_data': mm_data})
            valid.append(state)
        return prompts, valid

    def _process_output(
        self, state, vllm_output, dataset_name, sample_name, run_id, turn, save_zoom_images
    ):
        text = vllm_output.outputs[0].text
        state.vllm_outputs.append(vllm_output)
        state.turn_texts.append(text)

        tool_calls = self.tool_parser(text)
        if not tool_calls:
            state.messages.append({"role": "assistant", "content": text})
            state.is_finished = True
            return

        # Parse assistant turn: split at first tool_call tag
        first_pos = text.find('<tool_call>')
        thinking_part = text[:first_pos].strip() if first_pos > 0 else ""
        last_end = text.rfind('</tool_call>')
        tool_part = (
            text[first_pos: last_end + len('</tool_call>')]
            if last_end >= 0 else text[first_pos:]
        )
        content = []
        if thinking_part:
            content.append({"text": thinking_part})
        content.append({"text": ""})
        content.append({"text": tool_part})
        state.messages.append({"role": "assistant", "content": content})

        # Use per-trace sample_name if available (for cross-sample batching)
        effective_sample_name = state.sample_name if state.sample_name else sample_name

        # Execute tools
        tool_responses = []
        for idx, (t_name, t_args) in enumerate(tool_calls):
            call_id = turn * 100 + idx
            success, result_path, _, desc, bbox = self.tool_executor(
                tool_name=t_name,
                tool_args_json=t_args,
                all_image_paths=state.image_paths,
                dataset_name=dataset_name,
                sample_name=effective_sample_name,
                turn_count=call_id,
                run_id=run_id,
                trace_idx=state.trace_id,
                save_image=save_zoom_images,
            )
            if success and result_path:
                state.image_paths.append(result_path)
                tool_responses.append(result_path)
                if bbox:
                    state.tool_bboxes.append(bbox)
            else:
                state.is_finished = True
                return

        # Feed tool results back as user message
        user_content = []
        for path in tool_responses:
            user_content.append({"text": "<tool_response>\n"})
            user_content.append({"image": path})
            user_content.append({"text": "\n</tool_response>"})
        state.messages.append({"role": "user", "content": user_content})
