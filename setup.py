from setuptools import setup
import os
from setuptools import dist

required = [
    "numpy",
    "torch>=2.0.0",
    "transformers==4.57.6",
    "vllm==0.11.0",
    "accelerate>=0.20.0",
    "matplotlib>=3.5.0",
    "psutil",
    # `futures` is a Python-2-only backport of concurrent.futures (stdlib on
    # Python 3). Unconditional it breaks `pip install` on Python 3.12 with
    # "This backport is meant only for Python 2." Guard it to Python 2.
    "futures; python_version < '3'",
    # "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl",
    "black==22.8.0",
    "flake8==5.0.4",
    "datasets",
]


setup(
    name="es-at-scale",
    version="0.0.1",
    description="Python code to fine-tune LLMs with Evolution Strategies.",
    author="Cognizant AI Lab",
    packages=["es_at_scale"],
    install_requires=required,
    extras_require={
        # Math task grading (see es_at_scale/reward_function/math_grader.py).
        "math": ["math-verify", "pylatexenc", "latex2sympy2_extended"],
        # Experiment logging backend.
        "logging": ["wandb"],
    },
)
