"""
Utility functions for TTSP: answer extraction, evaluation, output processing,
and file saving.
"""

import os
import json
import math
from typing import List, Dict, Any, Optional

from .reliability import calculate_token_entropies


# ============= ANSWER EXTRACTION =============

def extract_answer(text: str) -> Optional[str]:
    """Extract the content of \\boxed{...} from model output."""
    if "boxed" not in text:
        return None
    ans = text.split("boxed")[-1]
    if not ans:
        return ""
    if ans[0] == "{":
        stack, a = 1, ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a.strip() or None


# ============= ANSWER COMPARISON =============

def equal_func(answer: str, ground_truth: str) -> bool:
    """Compare predicted answer to ground truth (case-insensitive for single letters)."""
    if not answer or not ground_truth:
        return False
    a, g = str(answer).strip(), str(ground_truth).strip()
    if len(a) == 1 and a.isalpha() and len(g) == 1 and g.isalpha():
        return a.upper() == g.upper()
    try:
        from dynasor.core.evaluator import math_equal
        return math_equal(a, g)
    except Exception:
        return a == g


# ============= OUTPUT PROCESSING =============

def extract_token_trace(token_ids, logprobs) -> List[Dict]:
    """Build a token-level trace list with id, probability, and decoded text."""
    result = []
    if not token_ids or not logprobs or len(token_ids) != len(logprobs):
        return result
    for tid, lp in zip(token_ids, logprobs):
        if lp is None:
            result.append({"id": tid, "prob": 0.0, "decode": ""})
            continue
        val = lp.get(tid)
        prob, decode = 0.0, ""
        if val is not None:
            log_val = getattr(val, 'logprob', val.get('logprob', -float('inf')) if isinstance(val, dict) else val)
            try:
                prob = math.exp(log_val)
            except OverflowError:
                prob = 0.0
            decode = getattr(val, 'decoded_token', None) or (val.get('decoded_token', '') if isinstance(val, dict) else '')
        result.append({"id": tid, "prob": prob, "decode": decode})
    return result


def process_trace_state(state) -> Dict[str, Any]:
    """
    Convert a completed TraceState into a trace dict for reliability analysis and voting.
    Combines token ids / logprobs across all turns.
    """
    all_ids, all_logprobs = [], []
    boundaries = []
    pos = 0

    for vllm_out in state.vllm_outputs:
        out = vllm_out.outputs[0]
        ids = list(out.token_ids or [])
        lps = list(out.logprobs or [])
        start, end = pos, pos + len(ids) - 1
        boundaries.append((start, end))
        all_ids.extend(ids)
        all_logprobs.extend(lps)
        pos += len(ids)

    token_entropies = calculate_token_entropies(all_logprobs)
    last_text = state.turn_texts[-1] if state.turn_texts else ""
    extracted_answer = extract_answer(last_text)

    # If generation ended due to length, discard the answer
    if state.vllm_outputs and state.vllm_outputs[-1].outputs[0].finish_reason == "length":
        extracted_answer = None

    return {
        "trace_id": state.trace_id,
        "texts": state.turn_texts,
        "token_ids": all_ids,
        "token_entropies": token_entropies,
        "token_trace": extract_token_trace(all_ids, all_logprobs),
        "num_tokens": len(all_ids),
        "extracted_answer": extracted_answer,
        "turn_boundaries": boundaries,
        "num_turns": len(state.turn_texts),
        "tool_bboxes": state.tool_bboxes,
        "stop_reason": (
            state.vllm_outputs[-1].outputs[0].finish_reason if state.vllm_outputs else "unknown"
        ),
        "is_fresh": state.is_fresh,
    }


# ============= EVALUATION =============

def evaluate_voting(voting_results: Dict, ground_truth: str) -> Dict[str, Dict]:
    """Check TTSP voting result against ground truth."""
    evaluation = {}
    res = voting_results.get("TTSP") if voting_results else None
    if res and res.get('answer') is not None:
        ans = str(res['answer']).strip()
        evaluation["TTSP"] = {
            'answer': ans,
            'is_correct': equal_func(ans, ground_truth),
            'num_votes': res.get('num_votes', 0),
        }
    else:
        evaluation["TTSP"] = {'answer': None, 'is_correct': False, 'num_votes': 0}
    return evaluation


def print_evaluation_report(question, ground_truth, evaluation, output):
    print(f"\n--- Evaluation ({output.total_traces_count} traces, {output.total_tokens} tokens) ---")
    print(f"GT: {ground_truth}")
    correct = sum(
        1 for t in output.all_traces
        if t.get('extracted_answer') and equal_func(t['extracted_answer'], ground_truth)
    )
    total = len(output.all_traces)
    if total:
        print(f"Trace Acc: {correct}/{total} ({correct/total:.1%})")
    
    ev = evaluation.get("TTSP", {})
    ans = str(ev.get('answer', 'N/A'))
    ans = (ans[:17] + '..') if len(ans) > 19 else ans
    print(f"{'Method':<25} {'Answer':<20} {'Correct':<8} {'Votes':<6}")
    print("-" * 65)
    print(f"{'TTSP':<25} {ans:<20} {'✓' if ev.get('is_correct') else '✗':<8} {ev.get('num_votes', 0):<6}")


# ============= SAVING =============

def print_summary(all_results: List[Dict], args):
    """Print evaluation summary with per-round accuracy and append to results JSONL.
    For MME-RealWorld-Lite, also print per-Task-SubTask accuracy."""
    valid = [r for r in all_results if 'error' not in r]
    if not valid:
        print("No valid results.")
        return

    # --- Final accuracy per method ---
    method_stats: Dict[str, Dict] = {}
    for r in valid:
        for m, ev in (r.get('evaluation') or {}).items():
            s = method_stats.setdefault(m, {'c': 0, 't': 0})
            s['t'] += 1
            if ev.get('is_correct'):
                s['c'] += 1

    # --- Per-round accuracy per method (cumulative voting at each round) ---
    round_stats: Dict[int, Dict[str, Dict]] = {}  # {round_idx: {method: {c, t}}}
    for r in valid:
        for rd_idx, rd_eval in (r.get('per_round_evaluation') or {}).items():
            rd_idx = int(rd_idx)
            rd_methods = round_stats.setdefault(rd_idx, {})
            for m, ev in rd_eval.items():
                s = rd_methods.setdefault(m, {'c': 0, 't': 0})
                s['t'] += 1
                if ev.get('is_correct'):
                    s['c'] += 1

    # --- Per-Task-SubTask accuracy for MME-RealWorld-Lite ---
    task_subtask_stats = {}
    if args.dataset == 'mme_realworld_lite':
        for r in valid:
            task = r.get('task', 'Unknown')
            subtask = r.get('subtask', 'Unknown')
            key = f"{task}/{subtask}"
            if key not in task_subtask_stats:
                task_subtask_stats[key] = {'task': task, 'subtask': subtask, 'methods': {}}
            for m, ev in (r.get('evaluation') or {}).items():
                s = task_subtask_stats[key]['methods'].setdefault(m, {'c': 0, 't': 0})
                s['t'] += 1
                if ev.get('is_correct'):
                    s['c'] += 1

    print(f"\n{'='*60}\nTTSP EVALUATION: {args.dataset.upper()} ({args.subset})\n{'='*60}")
    print(f"Total: {len(all_results)} | Valid: {len(valid)}")

    # Print per-round accuracy if there are multiple rounds
    if round_stats and len(round_stats) > 1:
        all_methods = sorted({m for rd in round_stats.values() for m in rd})
        header = f"{'Round':<10}" + "".join(f"{m:<25}" for m in all_methods)
        print(f"\n--- Per-Round Accuracy (cumulative) ---")
        print(header)
        print("-" * (10 + 25 * len(all_methods)))
        for rd_idx in sorted(round_stats):
            parts = [f"Round {rd_idx:<4}"]
            for m in all_methods:
                s = round_stats[rd_idx].get(m, {'c': 0, 't': 0})
                if s['t'] > 0:
                    parts.append(f"{s['c']}/{s['t']} ({s['c']/s['t']:.1%})")
                else:
                    parts.append("N/A")
            print(f"{parts[0]:<10}" + "".join(f"{p:<25}" for p in parts[1:]))

    # Print per-Task-SubTask accuracy for MME-RealWorld-Lite
    if task_subtask_stats:
        print(f"\n--- Per-Task-SubTask Accuracy ---")
        # Get all methods
        all_methods = sorted({m for ts in task_subtask_stats.values() for m in ts['methods']})
        print(f"{'Task/Subtask':<40}" + "".join(f"{m:<20}" for m in all_methods))
        print("-" * (40 + 20 * len(all_methods)))
        for key in sorted(task_subtask_stats.keys()):
            ts = task_subtask_stats[key]
            parts = [f"{key:<40}"]
            for m in all_methods:
                s = ts['methods'].get(m, {'c': 0, 't': 0})
                if s['t'] > 0:
                    parts.append(f"{s['c']}/{s['t']} ({s['c']/s['t']:.1%})")
                else:
                    parts.append("N/A")
            print("".join(f"{p:<20}" for p in parts))
        print("-" * (40 + 20 * len(all_methods)))

    # Print final accuracy
    print(f"\n--- Final Accuracy ---")
    print(f"{'Method':<25} {'Accuracy'}")
    print("-" * 45)
    for m, s in sorted(method_stats.items()):
        print(f"{m:<25} {s['c']}/{s['t']} ({s['c']/s['t']:.1%})")
    print("=" * 60)

    import datetime
    summary = {
        'dataset': args.dataset,
        'subset': args.subset,
        'model': args.model,
        'rounds': args.rounds,
        'budget_per_round': args.budget_per_round,
        'timestamp': datetime.datetime.now().isoformat(),
        'stats': {
            m: {'correct': s['c'], 'total': s['t'], 'acc': round(s['c'] / s['t'], 4)}
            for m, s in method_stats.items()
        },
        'per_round_stats': {
            str(rd_idx): {
                m: {'correct': s['c'], 'total': s['t'], 'acc': round(s['c'] / s['t'], 4)}
                for m, s in rd_methods.items()
            }
            for rd_idx, rd_methods in sorted(round_stats.items())
        },
    }
    # Add per-Task-SubTask stats for MME-RealWorld-Lite
    if task_subtask_stats:
        summary['per_task_subtask_stats'] = {
            key: {
                'task': ts['task'],
                'subtask': ts['subtask'],
                'methods': {
                    m: {'correct': s['c'], 'total': s['t'], 'acc': round(s['c'] / s['t'], 4)}
                    for m, s in ts['methods'].items()
                }
            }
            for key, ts in task_subtask_stats.items()
        }
    results_dir = os.path.join("results", args.model, args.subset)
    os.makedirs(results_dir, exist_ok=True)
    summary_file = os.path.join(results_dir, f"{args.dataset}_summary.jsonl")
    with open(summary_file, 'a') as f:
        json.dump(summary, f, ensure_ascii=False)
        f.write('\n')
    print(f"Summary saved: {summary_file}")


def save_trace_details(all_results: List[Dict], args):
    """Save per-trace details to JSONL for offline analysis."""
    if not args:
        return
    results_dir = os.path.join("results", args.model, args.subset)
    os.makedirs(results_dir, exist_ok=True)
    base = f"{args.dataset}_{args.subset}_r{args.rounds}_b{args.budget_per_round}"
    path = os.path.join(results_dir, f"{base}_0.jsonl")
    idx = 0
    while os.path.exists(path):
        idx += 1
        path = os.path.join(results_dir, f"{base}_{idx}.jsonl")

    with open(path, 'w') as f:
        for res in all_results:
            if 'error' in res:
                continue
            for i, t in enumerate(res.get('all_traces', [])):
                f.write(json.dumps({
                    'qid': res.get('qid'),
                    'question': res.get('question'),
                    'ground_truth': res.get('ground_truth'),
                    'idx': i,
                    'round_idx': t.get('round_idx'),
                    'pred_answer': t.get('extracted_answer'),
                    'turn_boundaries': t.get('turn_boundaries'),
                    'entropy': [
                        round(e, 6) if e is not None else None
                        for e in t.get('token_entropies', [])
                    ],
                    'evidence_ledger_input': t.get('evidence_ledger_input'),
                    # Compatibility key for initial-release analysis scripts.
                    'visual_facts_input': t.get('evidence_ledger_input'),
                }, ensure_ascii=False) + '\n')
    print(f"Traces saved: {path}")

    # Save debug details if requested
    if getattr(args, 'save_debug_details', False):
        debug_path = os.path.join(results_dir, f"{base}_{idx}_debug.jsonl")
        with open(debug_path, 'w') as f:
            for res in all_results:
                if 'error' in res:
                    continue
                # Get the TTSPOutput object and use to_debug_dict()
                output_obj = res.get('_output_obj')
                if output_obj and hasattr(output_obj, 'to_debug_dict'):
                    debug_data = output_obj.to_debug_dict()
                else:
                    # Fallback: construct from available data
                    debug_data = {
                        'qid': res.get('qid'),
                        'question': res.get('question'),
                        'ground_truth': res.get('ground_truth'),
                        'all_traces': res.get('all_traces', []),
                        'per_round_results': res.get('per_round_results', []),
                    }
                debug_data['qid'] = res.get('qid')
                debug_data['question'] = res.get('question')
                debug_data['ground_truth'] = res.get('ground_truth')
                f.write(json.dumps(debug_data, ensure_ascii=False) + '\n')
        print(f"Debug details saved: {debug_path}")
