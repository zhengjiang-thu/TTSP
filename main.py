"""
TTSP: Test-Time Scaling over Perception
Unified benchmark evaluation entry point.
"""

import os
import argparse
from vllm import SamplingParams

from TTSP import TTSPPipeline
from TTSP.config import get_sampling_params
from TTSP.dataload import (
    get_subset_info, load_dataset, load_sample, cleanup_sample, SUBSET_REGISTRY
)
from typing import List
from TTSP.utils import (
    evaluate_voting, print_summary, save_trace_details
)


def process_dataset(data_source, pipeline, args, actual_subset, dataset_type):
    all_results = []
    total = len(data_source)
    if args.max_questions and args.max_questions > 0:
        total = min(total, args.max_questions)

    # --- Load all samples upfront ---
    samples = []
    for qid in range(total):
        sample = load_sample(qid, dataset_type, data_source, actual_subset, args.dataset_path)
        sample['sample_name'] = f"{args.subset}/{qid}"
        sample['qid'] = qid
        samples.append(sample)

    print(f"Loaded {len(samples)} samples for batch processing")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        logprobs=10,
    )

    # --- Run cross-sample batch pipeline ---
    outputs = pipeline.run_batch(
        samples=samples,
        rounds=args.rounds,
        budget_per_round=args.budget_per_round,
        batch_size=args.batch_size,
        sampling_params=sampling_params,
        filtering_ratio=args.filtering_ratio,
        gamma=args.gamma,
        reasoning_effort=args.reasoning_effort,
        dataset_name=actual_subset,
        save_zoom_images=args.save_zoom_images,
        max_extraction_traces=args.max_extraction_traces,
        save_debug_details=args.save_debug_details,
        fresh_exploration_ratio=args.fresh_exploration_ratio,
        max_previous_knowledge_chars=args.max_previous_knowledge_chars,
    )

    # --- Per-sample evaluation ---
    for sample, output in zip(samples, outputs):
        ground_truth = sample['ground_truth']
        evaluation = (
            evaluate_voting(output.voting_results, ground_truth)
            if ground_truth and output.voting_results
            else None
        )

        # Per-round evaluation (cumulative voting at each round vs ground truth)
        per_round_eval = {}
        if ground_truth:
            for r in output.per_round_results:
                if r.voting_results:
                    per_round_eval[r.round_idx] = evaluate_voting(r.voting_results, ground_truth)

        cleanup_sample(dataset_type, sample['img_path'])

        result = {
            **output.to_dict(),
            '_output_obj': output,  # Keep reference for debug saving
            'question': sample['question'],
            'ground_truth': ground_truth,
            'qid': sample['qid'],
            'evaluation': evaluation,
            'per_round_evaluation': per_round_eval,
            'dataset': dataset_type,
            'subset': args.subset,
            'options': sample['options'],
        }

        # Add Task/Subtask info for MME-RealWorld-Lite
        if dataset_type == 'mme_realworld_lite':
            result['task'] = sample.get('task', '')
            result['subtask'] = sample.get('subtask', '')
            result['category'] = sample.get('category', '')

        all_results.append(result)

    return all_results


def get_dataset_subsets(dataset: str) -> List[str]:
    """Get all subsets for a given dataset."""
    subsets = []
    for subset, (_, ds_type, _) in SUBSET_REGISTRY.items():
        if ds_type == dataset:
            subsets.append(subset)
    return subsets


def main():
    parser = argparse.ArgumentParser(description='TTSP: Test-Time Scaling over Perception')

    # Model & dataset
    parser.add_argument('--model', type=str, default='Qwen3-VL-8B-Thinking')
    parser.add_argument('--model_dir', type=str, default='./model')
    parser.add_argument('--dataset', type=str, default='vstar',
                        choices=['vstar', 'hrbench', 'treebench', 'mme_realworld_lite'])
    parser.add_argument('--subset', type=str, default='Attr',
                        help='Subset name or "all" for all subsets of the dataset')
    parser.add_argument('--dataset_path', type=str, default='./dataset/vstar_bench')

    # TTSP-specific
    parser.add_argument('--rounds', type=int, default=4,
                        help='Number of iterative TTSP rounds (>=1)')
    parser.add_argument('--budget_per_round', type=int, default=8,
                        help='Parallel perception traces per round')
    parser.add_argument('--batch_size', type=int, default=1500,
                        help='vLLM batch size for trace generation (larger = better GPU utilization)')
    parser.add_argument('--filtering_ratio', type=float, default=0.4,
                        help='Fraction of low-reliability traces to discard per round (rho in paper)')
    parser.add_argument('--max_extraction_traces', type=int, default=100,
                        help='Max traces fed into knowledge extractor')
    parser.add_argument('--fresh_exploration_ratio', type=float, default=0.4,
                        help='Fraction of traces to generate fresh (without knowledge) in each round (alpha in paper)')
    parser.add_argument('--max_previous_knowledge_chars', type=int, default=20480,
                        help='Max characters of previous knowledge to include in extraction prompt')

    # Sampling
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=None,
                        help='Temperature for reliability-weighted voting (gamma in paper)')
    parser.add_argument('--reasoning_effort', type=str, default=None,
                        choices=['low', 'high', None])

    # Infrastructure
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95)
    parser.add_argument('--max_model_len', type=int, default=64000)
    parser.add_argument('--max_questions', type=int, default=None,
                        help='Debug mode: only process first N samples')
    parser.add_argument('--save_zoom_images', action='store_true', default=False)
    parser.add_argument('--save_debug_details', action='store_true', default=False,
                        help='Save full generated text and extracted knowledge for debugging')

    args = parser.parse_args()

    # Fill in defaults from model config
    model_cfg = get_sampling_params(args.model)
    for key in ('temperature', 'top_p', 'top_k', 'gamma', 'reasoning_effort'):
        if getattr(args, key) is None:
            setattr(args, key, model_cfg.get(key))

    args.model_path = os.path.join(args.model_dir, args.model)

    # Determine subsets to process
    if args.subset.lower() == 'all':
        subsets_to_process = get_dataset_subsets(args.dataset)
        if not subsets_to_process:
            raise ValueError(f"No subsets found for dataset '{args.dataset}'")
        print(f"\nWill process all {len(subsets_to_process)} subsets for {args.dataset}: {subsets_to_process}")
    else:
        if args.subset not in SUBSET_REGISTRY:
            raise ValueError(f"Unknown subset '{args.subset}'. Use 'all' or one of: {list(SUBSET_REGISTRY.keys())}")
        subsets_to_process = [args.subset]

    # Initialize pipeline once
    print("\n" + "="*80)
    print(f"TTSP Evaluation")
    print(f"Model:   {args.model}")
    print(f"Dataset: {args.dataset.upper()}")
    if args.subset.lower() == 'all':
        print(f"Subset:  ALL ({len(subsets_to_process)} subsets)")
    else:
        print(f"Subset:  {args.subset}")
    print(f"Rounds:  {args.rounds}  |  Budget/round: {args.budget_per_round}")
    if args.max_questions:
        print(f"DEBUG MODE: Processing only first {args.max_questions} samples per subset")
    if args.save_debug_details:
        print(f"DEBUG MODE: Saving full text and knowledge details")
    print("="*80)

    pipeline = TTSPPipeline(
        model=args.model_path,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    # Process all subsets
    all_results_combined = []
    for subset in subsets_to_process:
        print(f"\n{'='*80}")
        print(f"Processing subset: {subset}")
        print('='*80)

        # Temporarily override subset for this iteration
        original_subset = args.subset
        args.subset = subset

        actual_subset, dataset_type, category_filter = get_subset_info(subset)
        data_source, _ = load_dataset(dataset_type, args, actual_subset, category_filter)

        subset_results = process_dataset(data_source, pipeline, args, actual_subset, dataset_type)
        all_results_combined.extend(subset_results)

        if subset_results:
            print_summary(subset_results, args)
            save_trace_details(subset_results, args)

        # Restore original subset
        args.subset = original_subset

    # Final summary across all subsets
    if len(subsets_to_process) > 1 and all_results_combined:
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY: All {len(subsets_to_process)} subsets completed")
        print('='*80)
        print_summary(all_results_combined, args)


if __name__ == '__main__':
    main()
