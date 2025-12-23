"""
ISCAI 2025 Hardware Benchmark Script for NeuroSymGen.
Run this on A100/H100 to generate performance numbers.
**روی CPU هم کار می‌کنه (بدون نیاز به MSVC)**
"""

import torch
import sys
import os
import json

# Add parent directory to path for local testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neurosymgen import NeuroSymGenLayer, HardwareProfiler, HardwareOptimizer


def run_benchmark_suite():
    """Run complete hardware benchmark suite"""
    print("🚀 ISCAI 2025 NeuroSymGen Hardware Benchmark")
    print("==================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  Warning: CUDA not detected. Running on CPU.")
        print("    Set TORCH_COMPILE_DISABLE=1 to avoid compile errors\n")

    # Test configurations (کوچکتر برای CPU)
    # configs = [
    #     {"batch": 2, "input": 128, "rules": 4, "output": 64},
    #     {"batch": 4, "input": 256, "rules": 8, "output": 128},
    # ]
    #
    # if device == "cpu":
    #     configs = [{"batch": 4, "input": 128, "rules": 8, "output": 64}]
    configs = [
        {"batch": 16, "input": 512, "rules": 16, "output": 256},  # Real-world size
        {"batch": 32, "input": 1024, "rules": 32, "output": 512},  # Large model
        {"batch": 64, "input": 2048, "rules": 64, "output": 1024},  # ISCAI scale
    ]

    if device == "cpu":
        configs = [{"batch": 8, "input": 256, "rules": 16, "output": 128}]

    results = []

    for i, cfg in enumerate(configs):
        print(f"\n📊 Test {i + 1}/{len(configs)}: {cfg}")

        try:
            # Initialize model
            model = NeuroSymGenLayer(
                input_size=cfg["input"],
                num_rules=cfg["rules"],
                output_size=cfg["output"]
            ).to(device)

            # Optimize (روی CPU خودکار safe mode فعال می‌شه)
            optimizer = HardwareOptimizer(model, device=device)

            # Create sample inputs
            x = torch.randn(cfg["batch"], cfg["input"]).to(device)
            kg_data = torch.randn(cfg["batch"], 5, cfg["output"]).to(device)

            # Profile (بدون torch.compile روی CPU)
            profiler = HardwareProfiler(model, device=device)
            metrics = profiler.profile_full_forward(x, kg_data=kg_data)

            # Optimize with benchmark
            sample_input = x
            optimized_model = optimizer.optimize(sample_input)

            # Benchmark the optimized version
            bench_metrics = optimizer.benchmark({"x": x, "kg_data": kg_data})

            # Combine results
            metrics.update(cfg)
            metrics.update(bench_metrics)
            metrics["optimization_gain"] = (
                    (metrics["latency_ms"] - bench_metrics["latency_ms"]) / metrics["latency_ms"] * 100
            )

            results.append(metrics)

            print(f"  ✅ Latency: {metrics['latency_ms']:.2f} ms")
            print(f"  ✅ Optimized: {bench_metrics['latency_ms']:.2f} ms")
            print(f"  ✅ Gain: {metrics['optimization_gain']:.1f}%")

        except Exception as e:
            print(f"  ❌ Error in config {cfg}: {e}")
            print("     Skipping to next config...")
            continue

    # Save results
    if results:
        with open("iscai_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\n✅ Benchmark complete! Results saved to iscsi_results.json")

        # Summary
        avg_gain = sum(r.get("optimization_gain", 0) for r in results) / len(results)
        print(f"\n📈 Average optimization gain: {avg_gain:.1f}%")
    else:
        print("\n❌ No successful benchmarks completed")


if __name__ == "__main__":
    # تنظیمات محیطی برای جلوگیری از compile روی CPU
    if not torch.cuda.is_available():
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        print("🔧 TORCH_COMPILE_DISABLE=1 set for CPU execution")

    run_benchmark_suite()