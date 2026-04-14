"""
TTSP: Test-Time Scaling over Perception
========================================

A framework for addressing the Grounding Paradox in tool-augmented visual reasoning.

The Grounding Paradox refers to the circular dependency where a model must decide 
where to look in an image before it has access to the fine-grained information 
needed to make that decision correctly.

TTSP resolves this by treating perception itself as a scalable inference process:

1. **Parallel Perceptual Exploration**: Generate K independent perception traces 
   that explore different regions of the image using visual tools (zoom).

2. **Reliability Filtering**: Score each trace using token-level entropy and 
   filter out unreliable traces.

3. **Knowledge Extraction**: Distill filtered traces into structured knowledge 
   (Confirmed Knowledge + Open Conflicts).

4. **Iterative Refinement**: Feed extracted knowledge back to guide subsequent 
   rounds of exploration.

5. **Weighted Voting**: Aggregate answers from all reliable traces using 
   reliability-weighted voting.

Example:
    >>> from TTSP import TTSPPipeline
    >>> from vllm import SamplingParams
    >>> 
    >>> pipeline = TTSPPipeline(model="path/to/Qwen3-VL-8B-Instruct")
    >>> sampling_params = SamplingParams(temperature=0.6, logprobs=10)
    >>> 
    >>> output = pipeline.run(
    ...     question="What color is the car?",
    ...     options=["Red", "Blue", "Green", "Yellow"],
    ...     image_path="image.jpg",
    ...     rounds=4,
    ...     budget_per_round=8,
    ...     sampling_params=sampling_params,
    ... )
    >>> print(output.voting_results['TTSP']['answer'])

For more information, see the paper:
"Test-time Scaling over Perception: Resolving the Grounding Paradox in Thinking with Images"
"""

from .pipeline import TTSPPipeline
from .outputs import TTSPOutput, PerRoundResult

__version__ = "1.0.0"
__author__ = "Zheng Jiang"
__email__ = "jz24@mails.tsinghua.edu.cn"

__all__ = ["TTSPPipeline", "TTSPOutput", "PerRoundResult"]
