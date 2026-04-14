"""
Model configuration and sampling parameter defaults for TTSP.
"""
import re

MODEL_TYPE_CONFIG = {
    "thinking": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "gamma": 0.1},
    "instruct": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "gamma": 1.0},
}


def _extract_model_size(model_name: str) -> int:
    match = re.search(r'(\d+)[Bb](?:-|_)', model_name)
    return int(match.group(1)) if match else 0


def get_sampling_params(model_name: str) -> dict:
    """Get default sampling parameters for the given model name."""
    model_type = "thinking" if "thinking" in model_name.lower() else "instruct"
    config = MODEL_TYPE_CONFIG[model_type].copy()
    if _extract_model_size(model_name) >= 32:
        config["temperature"] = 1.0
    return config
