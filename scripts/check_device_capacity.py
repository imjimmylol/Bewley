#!/usr/bin/env python3
"""
Check device memory capacity for PER buffer sizing.
Run: python scripts/check_device_capacity.py
"""

import torch
import psutil
import sys

def bytes_to_gb(bytes_val):
    return bytes_val / (1024 ** 3)

def check_capacity():
    print("=" * 60)
    print("DEVICE CAPACITY CHECK FOR PER BUFFER SIZING")
    print("=" * 60)

    # === CPU / RAM ===
    ram = psutil.virtual_memory()
    print(f"\n[RAM]")
    print(f"  Total:     {bytes_to_gb(ram.total):.2f} GB")
    print(f"  Available: {bytes_to_gb(ram.available):.2f} GB")
    print(f"  Used:      {bytes_to_gb(ram.used):.2f} GB ({ram.percent}%)")

    # === GPU (CUDA) ===
    if torch.cuda.is_available():
        print(f"\n[CUDA GPU]")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory
            reserved = torch.cuda.memory_reserved(i)
            allocated = torch.cuda.memory_allocated(i)
            free = total - reserved

            print(f"  GPU {i}: {props.name}")
            print(f"    Total:     {bytes_to_gb(total):.2f} GB")
            print(f"    Reserved:  {bytes_to_gb(reserved):.2f} GB")
            print(f"    Allocated: {bytes_to_gb(allocated):.2f} GB")
            print(f"    Free:      {bytes_to_gb(free):.2f} GB")
    else:
        print(f"\n[CUDA GPU]: Not available")

    # === MPS (Apple Silicon) ===
    if torch.backends.mps.is_available():
        print(f"\n[MPS (Apple Silicon)]")
        print(f"  MPS is available")
        print(f"  Note: MPS shares unified memory with RAM")
        print(f"  Recommended: Use ~50% of available RAM for buffer")
    else:
        print(f"\n[MPS]: Not available")

    # === Estimate buffer capacity ===
    print("\n" + "=" * 60)
    print("BUFFER SIZE ESTIMATION")
    print("=" * 60)

    # Your config: batch_size=64, n_agents=800
    batch_size = 64
    n_agents = 800

    # Memory per experience (storing minimal MainState inputs)
    # Per agent: savings(4) + ability(4) + moneydisposable(4) + ret(4) +
    #            is_superstar_vA(1) + is_superstar_vB(1) = 18 bytes (float32 + bool)
    # With float16 quantization: ~10 bytes/agent

    bytes_per_agent_f32 = 4 * 4 + 2  # 4 floats + 2 bools = 18 bytes
    bytes_per_agent_f16 = 2 * 4 + 2  # 4 float16s + 2 bools = 10 bytes

    bytes_per_exp_f32 = bytes_per_agent_f32 * n_agents + 8  # +8 for priority/step
    bytes_per_exp_f16 = bytes_per_agent_f16 * n_agents + 8

    print(f"\n[Per Experience Memory]")
    print(f"  Config: batch_size={batch_size}, n_agents={n_agents}")
    print(f"  float32: {bytes_per_exp_f32 / 1024:.2f} KB per experience")
    print(f"  float16: {bytes_per_exp_f16 / 1024:.2f} KB per experience")

    # Recommend buffer sizes based on available memory
    available_gb = bytes_to_gb(ram.available)

    # Reserve some memory for model, training, etc.
    buffer_budget_gb = available_gb * 0.3  # Use 30% of available RAM for buffer
    buffer_budget_bytes = buffer_budget_gb * (1024 ** 3)

    capacity_f32 = int(buffer_budget_bytes / bytes_per_exp_f32)
    capacity_f16 = int(buffer_budget_bytes / bytes_per_exp_f16)

    print(f"\n[Recommended Buffer Capacity]")
    print(f"  Memory budget (30% of available): {buffer_budget_gb:.2f} GB")
    print(f"  With float32: {capacity_f32:,} experiences")
    print(f"  With float16: {capacity_f16:,} experiences")

    # Sum tree overhead
    sumtree_overhead_f32 = capacity_f32 * 2 * 8 / (1024**3)  # 2*capacity nodes, 8 bytes each
    sumtree_overhead_f16 = capacity_f16 * 2 * 8 / (1024**3)

    print(f"\n[Sum Tree Overhead]")
    print(f"  With float32 capacity: {sumtree_overhead_f32:.3f} GB")
    print(f"  With float16 capacity: {sumtree_overhead_f16:.3f} GB")

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    # Round to nice numbers
    recommended_f16 = min(capacity_f16, 500_000)  # Cap at 500k for practicality
    recommended_f16 = (recommended_f16 // 10000) * 10000  # Round to nearest 10k

    print(f"\n  Suggested buffer_size: {recommended_f16:,}")
    print(f"  Storage type: float16 (recommended)")
    print(f"  Estimated memory: {recommended_f16 * bytes_per_exp_f16 / (1024**3):.2f} GB")

    return {
        "available_ram_gb": available_gb,
        "recommended_capacity": recommended_f16,
        "bytes_per_exp_f16": bytes_per_exp_f16,
    }

if __name__ == "__main__":
    check_capacity()
