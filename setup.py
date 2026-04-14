"""
Setup script for TTSP (Test-Time Scaling over Perception).
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies with flexible version constraints
# For exact versions used in development, see requirements.txt
CORE_DEPENDENCIES = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "transformers>=4.40.0",
    "qwen-vl-utils>=0.0.8",
    "vllm>=0.5.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "pillow>=10.0.0",
    "datasets>=2.14.0",
    "tqdm>=4.65.0",
]

setup(
    name="ttsp",
    version="1.0.0",
    author="Zheng Jiang",
    author_email="jz24@mails.tsinghua.edu.cn",
    description="Test-Time Scaling over Perception: Resolving the Grounding Paradox in Thinking with Images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ttsp",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=CORE_DEPENDENCIES,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
)
