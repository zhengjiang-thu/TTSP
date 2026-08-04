<div align="center">
  <img src="figs/logo.png" alt="TTSP logo" height="150">
  <h1>Test-Time Scaling over Perception</h1>
  <p><strong>Resolving the Grounding Paradox in Thinking with Images</strong></p>
  <a href="https://github.com/jiangz20/TTSP">
    <img src="https://img.shields.io/badge/GitHub-jiangz20%2FTTSP-black?logo=github" alt="GitHub repository">
  </a>
  <a href="https://arxiv.org/abs/2604.11025">
    <img src="https://img.shields.io/badge/arXiv-2604.11025-b31b1b.svg" alt="arXiv paper">
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="Apache 2.0 license">
  </a>
</div>

## Overview

TTSP is a training-free framework for scaling visual perception at test time. It addresses the **Grounding Paradox**: a model must decide where to inspect before it has acquired the visual evidence needed to make that decision reliably.

![Grounding Paradox](figs/introduction.png)

The current implementation follows the latest paper in four stages:

1. **Dual-pool coverage.** Round 1 samples all `K` traces independently. Later rounds sample `ceil(alpha * K)` fresh traces and condition the remainder on the Evidence Ledger.
2. **Round-relative selection.** Each trace is scored by the negative mean of its highest-entropy token positions. Exactly `ceil((1-rho) * K)` traces survive each round.
3. **Evidence-guided refinement.** A deterministic update maintains bounded **Confirmed Knowledge** and **Open Conflicts** tiers using Retain, Demote, Promote, and Confirm rules.
4. **Reliability-weighted readout.** Every trace retained in any round votes once with weight `exp(score / gamma)`; the accumulated pool is not filtered again.

![TTSP framework](figs/framework.png)

## Installation

```bash
git clone https://github.com/jiangz20/TTSP.git
cd TTSP

# Reproduce the development environment.
pip install -r requirements.txt

# Or install the package and its core dependencies.
pip install -e .
```

The released benchmark configuration uses vLLM and multiple GPUs. Adjust `tensor_parallel_size` and `batch_size` to fit your hardware.

## Quick start

```python
from TTSP import TTSPPipeline
from vllm import SamplingParams

pipeline = TTSPPipeline(
    model="path/to/Qwen3-VL-8B-Instruct",
    tensor_parallel_size=8,
    max_logprobs=256,
    gpu_memory_utilization=0.95,
    max_model_len=51200,
)

sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    top_k=0,       # disabled
    max_tokens=4096,
    logprobs=256,  # required by truncated-entropy scoring
)

output = pipeline.run(
    question="What is the color of the car in the image?",
    options=["Red", "Blue", "Green", "Yellow"],
    image_path="path/to/image.jpg",
    rounds=4,                    # N
    budget_per_round=8,          # K
    filtering_ratio=0.4,         # rho
    fresh_exploration_ratio=0.4, # alpha
    entropy_top_k=256,           # k_top
    critical_positions=128,      # k'
    gamma=1.0,
    max_ledger_traces=4,         # m
    confirmed_ledger_cap=8,
    conflict_ledger_cap=4,
    sampling_params=sampling_params,
)

print(output.voting_results["TTSP"]["answer"])
print(output.total_trace_tokens, output.total_ledger_tokens)
```

The default algorithm settings are `N=4`, `K=8`, `alpha=0.4`, `rho=0.4`, `k_top=256`, `k'=128`, `gamma=1.0`, ledger context size `m=4`, confirmed-tier cap `8`, and conflict-tier cap `4`. Trace generation uses temperature `1.0`, top-p `1.0`, and top-k disabled. Ledger updates use greedy decoding.

### Batch inference

```python
samples = [
    {
        "question": "Question 1?",
        "options": ["A", "B", "C", "D"],
        "image_path": "path/to/image1.jpg",
        "sample_name": "sample_1",
    },
    {
        "question": "Question 2?",
        "options": ["A", "B", "C", "D"],
        "image_path": "path/to/image2.jpg",
        "sample_name": "sample_2",
    },
]

outputs = pipeline.run_batch(
    samples=samples,
    rounds=4,
    budget_per_round=8,
    batch_size=1000,
)
```

## Benchmark CLI

```bash
python main.py \
    --model Qwen3-VL-8B-Instruct \
    --model_dir ./models \
    --tensor_parallel_size 8 \
    --dataset vstar \
    --subset Attr \
    --dataset_path ./dataset/vstar_bench \
    --rounds 4 \
    --budget_per_round 8 \
    --fresh_exploration_ratio 0.4 \
    --filtering_ratio 0.4
```

Supported dataset identifiers are `vstar`, `hrbench`, `treebench`, and `mme_realworld_lite`. Use `--subset all` to evaluate every registered subset of one dataset.

## Output semantics

`TTSPOutput` exposes:

- `all_traces`: every raw trace generated across rounds;
- `all_filtered_traces`: the union of the traces retained by per-round gating;
- `per_round_results`: trace, ledger, vote, token, and timing data for each round;
- `voting_results["TTSP"]`: the final reliability-weighted answer;
- `total_trace_tokens`, `total_ledger_tokens`, and `total_tokens`: generated-token accounting, including ledger-update overhead.

The initial release's `visual_facts`, `max_extraction_traces`, and related result keys remain as deprecated compatibility aliases.

## Project structure

```text
TTSP/
├── TTSP/
│   ├── pipeline.py             # Coverage-selection-utilization pipeline
│   ├── inference.py            # Multi-turn batched visual-tool inference
│   ├── reliability.py          # Truncated entropy and exact relative gating
│   ├── evidence_ledger.py      # Dual-tier ledger updates
│   ├── knowledge_extractor.py  # Backward-compatible wrapper
│   ├── voting.py               # Reliability-weighted retained-pool vote
│   ├── prompts.py              # Fresh, guided, and ledger-update prompts
│   ├── tools.py                # Image crop tool
│   ├── outputs.py              # Result data classes
│   ├── dataload.py             # Benchmark loaders
│   └── config.py               # Shared defaults
├── tests/                      # Lightweight semantic regression tests
├── main.py                     # Benchmark entry point
├── requirements.txt            # Reproducibility environment
└── setup.py
```

Run the lightweight tests with:

```bash
python -m unittest discover -s tests -v
```

## Paper and citation

The paper is available on arXiv: [Test-Time Scaling over Perception: Resolving the Grounding Paradox in Thinking with Images](https://arxiv.org/abs/2604.11025).

```bibtex
@article{jiang2026test,
  title={Test-Time Scaling over Perception: Resolving the Grounding Paradox in Thinking with Images},
  author={Jiang, Zheng and Chen, Yiming and He, Nan and Chen, Jiahui and Li, Chaoyang and Qian, Houde and Sun, Lifeng},
  journal={arXiv preprint arXiv:2604.11025},
  year={2026}
}
```

## License

TTSP is released under the [Apache License 2.0](LICENSE).

## Acknowledgments

This project builds on [RTWI](https://github.com/XLearning-SCU/Reliable_TWI), [DeepConf](https://github.com/facebookresearch/deepconf), and [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL).
