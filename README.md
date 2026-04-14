<div align="center">
  <img src="figs/logo.png" alt="logo" height="150">
  <h1 style="font-size: 32px; font-weight: bold;"> Test-time Scaling over Perception: Resolving the Grounding Paradox in Thinking with Images </h1>
  <a href="https://arxiv.org/pdf/2604.11025">
    <img src="https://img.shields.io/badge/ArXiv-TTSP-brown?logo=arxiv" alt="Paper">
  </a>
  <a href="https://github.com/zhengjiang-thu/TTSP">
    <img src="https://img.shields.io/badge/ Github-Homepage-black?logo=github" alt="Code">
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/📃 License-Apache%202.0-blue.svg" alt="License">
  </a>
</div>

## 🔥 News
+ **[2026.04.14]** 🎉 We have released the paper and code for TTSP.


## 📖 Overview

TTSP (Test-Time Scaling over Perception) is a framework that addresses the **Grounding Paradox** in tool-augmented visual reasoning. The Grounding Paradox refers to the circular dependency where models must decide where to look before they have access to the evidence needed to make that decision correctly.
![](figs/introduction.jpg)

### Key Features

- We identify the **Grounding Paradox** as a fundamental obstacle in tool-augmented visual reasoning, and show that it provides a principled explanation for the characteristic failure modes of current Thinking with Images systems.
- We introduce **TTSP**, a principled framework for test-time scaling over perception that enables multimodal models to explore, validate, and iteratively refine visual evidence under uncertainty, rather than relying on a single potentially misgrounded trajectory.
- We empirically demonstrate that **TTSP** consistently outperforms existing baselines across diverse visual reasoning benchmarks, providing a stronger foundation for advancing the Thinking with Images paradigm.
  
![](figs/framework.png)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/zhengjiang-thu/TTSP.git
cd TTSP

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

```python
from TTSP import TTSPPipeline
from vllm import SamplingParams

# Initialize the pipeline
pipeline = TTSPPipeline(
    model="path/to/Qwen3-VL-8B-Instruct",
    gpu_memory_utilization=0.95,
    max_model_len=64000,
)

# Define sampling parameters
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    max_tokens=4096,
    logprobs=10,
)

# Run TTSP inference
output = pipeline.run(
    question="What is the color of the car in the image?",
    options=["Red", "Blue", "Green", "Yellow"],
    image_path="path/to/image.jpg",
    rounds=4,                    # Number of iterative rounds (N)
    budget_per_round=8,          # Parallel traces per round (K)
    sampling_params=sampling_params,
    filtering_ratio=0.4,         # Fraction of traces to filter (ρ)
    gamma=1.0,                   # Temperature for weighted voting (γ)
    fresh_exploration_ratio=0.4, # Fraction of fresh traces (α)
)

# Get the final answer
print(f"Answer: {output.voting_results['TTSP']['answer']}")
```

### Batch Processing

```python
# Prepare samples
samples = [
    {
        "question": "Question 1?",
        "options": ["A", "B", "C", "D"],
        "image_path": "path/to/image1.jpg",
        "sample_name": "sample_1",
    },
    # ... more samples
]

# Run batch inference
outputs = pipeline.run_batch(
    samples=samples,
    rounds=4,
    budget_per_round=8,
    batch_size=1000,  # Adjust based on GPU memory
)

for sample, output in zip(samples, outputs):
    print(f"{sample['sample_name']}: {output.voting_results['TTSP']['answer']}")
```

## 🖥️ Command-Line Interface

### Running TTSP on Benchmarks
```bash
# V* Bench - Attribute subset
python main.py \
    --model Qwen3-VL-8B-Instruct \
    --model_dir ./models \
    --dataset vstar \
    --subset Attr \
    --dataset_path ./dataset/vstar_bench \
    --rounds 4 \
    --budget_per_round 8 \
    --filtering_ratio 0.4 \
    --fresh_exploration_ratio 0.4

# HR-Bench 4K - Fine-grained Single Perception
python main.py \
    --model Qwen3-VL-8B-Instruct \
    --dataset hrbench \
    --subset HR4K-FSP \
    --dataset_path ./dataset/hr_bench \
    --rounds 4 \
    --budget_per_round 8

# Process all subsets of a dataset
python main.py \
    --model Qwen3-VL-8B-Instruct \
    --dataset vstar \
    --subset all \
    --dataset_path ./dataset/vstar_bench
```

## 📊 Supported Datasets

- **Vstar Bench**: [Vstar Bench](https://huggingface.co/datasets/craigwu/vstar_bench)
- **HR-Bench**: [HR-bench](https://huggingface.co/datasets/DreamMr/HR-Bench)
- **TreeBench**: [TreeBench](https://huggingface.co/datasets/HaochenWang/TreeBench)
- **MME-RealWorld-Lite**: [MME-RealWorld-Lite](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lite)

## 💼 Project Structure

```
TTSP/
├── TTSP/                      # Main package
│   ├── __init__.py            # Package exports
│   ├── pipeline.py            # Core TTSP pipeline (TTSPPipeline)
│   ├── inference.py           # Multi-turn batch inference loop
│   ├── reliability.py         # Entropy-based reliability scoring
│   ├── knowledge_extractor.py # Structured knowledge extraction
│   ├── voting.py              # Voting aggregation strategies
│   ├── tools.py               # Visual tool implementation
│   ├── prompts.py             # System prompts for LLM
│   ├── outputs.py             # Output data classes
│   ├── utils.py               # Utility functions
│   ├── dataload.py            # Dataset loaders
│   └── config.py              # Model configuration
├── main.py                    # Benchmark evaluation entry point
├── baseline.py                # Single-pass baseline
├── requirements.txt           # Python dependencies
└──  setup.py                  # Package setup
```

## 📚 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{jiang2026ttsp,
  title={Test-time Scaling over Perception: Resolving the Grounding Paradox in Thinking with Images},
  author={Jiang, Zheng and Chen, Yiming and He, Nan and Chen, Jiahui and Li, Chaoyang and Qian, Houde and Sun, Lifeng},
  booktitle={Proceedings of the ACM Conference},
  year={2026}
}
```

## 📄 License

This project is released under [Apache licence](./LICENSE).

## 🙏 Acknowledgments

We would like to thank the following repos for their great work:
- This work is built upon the [RTWI](https://github.com/XLearning-SCU/Reliable_TWI) and [DeepConf](https://github.com/facebookresearch/deepconf).
- This work utilizes models from [Qwen](https://github.com/QwenLM/Qwen3-VL).
