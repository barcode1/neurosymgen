import torch
import pytest
from neurosymgen import NeuroSymGenLayer, HardwareProfiler


def test_hardware_efficiency():
    """Test hardware efficiency metrics"""
    layer = NeuroSymGenLayer(input_size=256, num_rules=10, output_size=128)
    layer = layer.to("cuda" if torch.cuda.is_available() else "cpu")

    profiler = HardwareProfiler(layer)

    x = torch.randn(8, 256).to(layer.device)
    kg_data = torch.randn(8, 10, 128).to(layer.device)

    metrics = profiler.profile_full_forward(x, kg_data)

    # ISCAI requires these metrics
    assert "latency_ms" in metrics
    assert "hbm_max_allocated_gb" in metrics
    assert metrics["latency_ms"] > 0
    assert metrics["hbm_max_allocated_gb"] > 0

    print(f"✅ Latency: {metrics['latency_ms']:.2f} ms")
    print(f"✅ HBM: {metrics['hbm_max_allocated_gb']:.2f} GB")
    print(f"✅ GFLOPs: {metrics.get('gflops', 0):.2f}")


if __name__ == "__main__":
    test_hardware_efficiency()