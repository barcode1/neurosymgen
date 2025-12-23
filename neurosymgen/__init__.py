"""
NeuroSymGen: Hardware-Optimized Neuro-Symbolic Framework v1.0.0
ISCAI 2025 Submission Ready
"""
from .core import NeuroSymGenLayer
from .hardware.profiler import HardwareProfiler
from .hardware.optimizer import HardwareOptimizer
from .reasoning.grok_client import GrokReActAgent
from .reasoning.validator import SymbolicValidator
from .kg.integrator import HeteroKGIntegrator
from .generation.generator import MultimodalStoryGenerator
from .utils.config import load_config

__version__ = "1.0.0-iscai"
__all__ = [
    "NeuroSymGenLayer", "HardwareProfiler", "HardwareOptimizer",
    "GrokReActAgent", "SymbolicValidator", "HeteroKGIntegrator",
    "MultimodalStoryGenerator", "load_config"
]

# Hardware compatibility check
import torch
if not torch.cuda.is_available():
    import warnings
    warnings.warn("CUDA not detected. ISCAI benchmarks require GPU.")