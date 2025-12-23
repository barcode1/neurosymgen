"""
Real GPU Benchmark for ISCAI 2025.
Run on A100/H100 to get publication-worthy numbers.
"""

import os
import json

os.environ.pop("TORCH_COMPILE_DISABLE", None)  # Enable compile

import torch
from neurosymgen import NeuroSymGenLayer, HardwareProfiler, HardwareOptimizer


def run_gpu_benchmark():
    """Run benchmarks that ISCAI expects"""
    assert torch.cuda.is_available(), "❌ GPU required for ISCAI"

    device = "cuda"
    print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ISCAI-scale configurations
    configs = [
        {"batch": 32, "input": 1024, "rules": 64, "output": 512, "name": "medium"},
        {"batch": 64, "input": 2048, "rules": 128, "output": 1024, "name": "large"},
        {"batch": 128, "input": 4096, "rules": 256, "output": 2048, "name": "xl"},
    ]

    results = []

    for cfg in configs:
        print(f"\n📊 {cfg['name'].upper()} config: {cfg}")

        # Create model
        model = NeuroSymGenLayer(
            input_size=cfg["input"],
            num_rules=cfg["rules"],
            output_size=cfg["output"]
        ).cuda()

        # Create data
        x = torch.randn(cfg["batch"], cfg["input"]).cuda()
        kg_data = torch.randn(cfg["batch"], 10, cfg["output"]).cuda()

        # Profile original
        profiler = HardwareProfiler(model, device=device)
        base_metrics = profiler.profile_full_forward(x, kg_data=kg_data)

        # Optimize with TensorRT
        optimizer = HardwareOptimizer(model, device=device)
        optimized_model = optimizer.optimize(x)

        # Benchmark optimized
        opt_metrics = optimizer.benchmark({"x": x, "kg_data": kg_data})

        # Calculate improvements
        results.append({
            "config": cfg["name"],
            "baseline_latency": base_metrics["latency_ms"],
            "optimized_latency": opt_metrics["latency_ms"],
            "speedup": base_metrics["latency_ms"] / opt_metrics["latency_ms"],
            "baseline_memory": base_metrics["hbm_max_allocated_gb"],
            "optimized_memory": opt_metrics["memory_gb"],
            "memory_saving": (
                    (base_metrics["hbm_max_allocated_gb"] - opt_metrics["memory_gb"]) /
                    base_metrics["hbm_max_allocated_gb"] * 100
            )
        })

    # Print summary
    print("\n" + "=" * 60)
    print("📈 ISCAI 2025 Performance Summary")
    print("=" * 60)

    for r in results:
        print(f"\n{r['config'].upper()}:")
        print(f"  Latency: {r['baseline_latency']:.2f} ms → {r['optimized_latency']:.2f} ms")
        print(f"  Speedup: {r['speedup']:.2f}x")
        print(f"  Memory efficiency: {r['memory_saving']:.1f}%")

    # Save for paper
    with open("iscai_gpu_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ GPU benchmark complete! Results saved.")


if __name__ == "__main__":
    run_gpu_benchmark()