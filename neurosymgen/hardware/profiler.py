import torch
import torch.profiler
from typing import Dict, Any, Optional
import time
import psutil
import os


class HardwareProfiler:
    """
    Hardware Profiler for ISCAI 2025 experiments.
    Compatible with PyTorch 2.0+
    """

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.metrics = {}

        if device == "cuda" and torch.cuda.is_available():
            self.gpu_device = torch.cuda.current_device()
        else:
            self.gpu_device = None

    def profile_full_forward(self, *args, **kwargs) -> Dict[str, float]:
        """Profile complete forward pass with all hardware metrics"""
        # Warmup
        for _ in range(3):
            _ = self.model(*args, **kwargs)

        if self.device == "cuda":
            return self._profile_cuda(*args, **kwargs)
        else:
            return self._profile_cpu(*args, **kwargs)

    def _profile_cuda(self, *args, **kwargs) -> Dict[str, float]:
        """Profile on NVIDIA GPUs with detailed metrics"""
        # Reset CUDA stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CUDA,
                    torch.profiler.ProfilerActivity.CPU
                ],
                record_shapes=True,
                with_stack=True,
                with_flops=True
        ) as prof:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            output = self.model(*args, **kwargs)
            end_event.record()

            torch.cuda.synchronize()

        # ⭐ FIX: استفاده از self_device_time_total برای PyTorch 2.x
        kernel_stats = prof.key_averages()

        # Calculate kernel time with fallback
        try:
            # PyTorch 2.x+ (recommended)
            kernel_time_us = sum(k.self_device_time_total for k in kernel_stats)
            kernel_time_ms = kernel_time_us / 1e3
        except AttributeError:
            # PyTorch 1.x fallback
            try:
                kernel_time_ms = sum(k.self_cuda_time_total for k in kernel_stats) / 1e6
            except AttributeError:
                # Last resort: estimate
                kernel_time_ms = 0

        # ⭐ FIX: محاسبه FLOPs با حفاظت
        try:
            flops = sum(k.flops for k in kernel_stats) if hasattr(kernel_stats[0], 'flops') else 0
        except:
            flops = 0

        latency_ms = start_event.elapsed_time(end_event)
        max_hbm_gb = torch.cuda.max_memory_allocated() / 1e9
        reserved_hbm_gb = torch.cuda.max_memory_reserved() / 1e9

        total_io_mb = self._estimate_pcie_transfer(*args, **kwargs)

        self.metrics = {
            "latency_ms": latency_ms,
            "kernel_time_ms": kernel_time_ms,
            "hbm_max_allocated_gb": max_hbm_gb,
            "hbm_reserved_gb": reserved_hbm_gb,
            "pcie_transfer_mb": total_io_mb,
            "gflops": flops / 1e9 if flops else 0,
            "memory_efficiency": max_hbm_gb / reserved_hbm_gb if reserved_hbm_gb > 0 else 0
        }

        return self.metrics

    def _profile_cpu(self, *args, **kwargs) -> Dict[str, float]:
        """Profile on CPU with memory tracking"""
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1e9

        start_time = time.perf_counter()
        output = self.model(*args, **kwargs)
        end_time = time.perf_counter()

        mem_after = process.memory_info().rss / 1e9

        self.metrics = {
            "latency_ms": (end_time - start_time) * 1000,
            "cpu_memory_gb": mem_after - mem_before,
            "cpu_percent": process.cpu_percent()
        }

        return self.metrics

    def _estimate_pcie_transfer(self, *args, **kwargs) -> float:
        """Estimate data transferred via PCIe"""
        total_bytes = 0

        for arg in args:
            if isinstance(arg, torch.Tensor):
                total_bytes += arg.numel() * arg.element_size()

        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                total_bytes += v.numel() * v.element_size()

        for p in self.model.parameters():
            total_bytes += p.numel() * p.element_size()

        return total_bytes / 1e6  # MB

    def compare_with_baseline(self, baseline_metrics: Dict[str, float]) -> Dict[str, float]:
        """Compare performance improvement vs baseline"""
        improvements = {}
        for key in self.metrics:
            if key in baseline_metrics and baseline_metrics[key] > 0:
                improvements[f"{key}_improvement"] = (
                        (baseline_metrics[key] - self.metrics[key]) / baseline_metrics[key] * 100
                )
        return improvements

    def export_for_snowscope(self, filename: str):
        """Export to Snowscope format for ISCAI visualization"""
        import json
        data = {
            "model": self.model.__class__.__name__,
            "metrics": self.metrics,
            "timestamp": time.time()
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)