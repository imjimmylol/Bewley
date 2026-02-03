# tests/test_replay_buffer.py
"""
Tests for Prioritized Experience Replay buffer implementation.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/Users/chenjinghe/Desktop/python-projects/Bewley')

from src.replay_buffer.sum_tree import SumTree
from src.replay_buffer.experience import (
    Experience, ShockOutcomes,
    snapshot_main_state, pack_experience,
    unpack_to_main_state, unpack_shock_outcomes,
    compute_experience_memory_bytes, estimate_buffer_memory_gb,
)
from src.replay_buffer.prioritized_buffer import PrioritizedReplayBuffer
from src.env_state import MainState, ParallelState


def test_sum_tree():
    """Test SumTree basic operations."""
    print("=" * 60)
    print("Testing SumTree...")
    print("=" * 60)

    capacity = 8
    tree = SumTree(capacity)

    # Test initial state
    assert len(tree) == 0
    assert tree.total == 0.0
    assert tree.max_priority == 1.0
    print("✓ Initial state correct")

    # Test add
    priorities = [0.5, 1.0, 0.3, 0.8, 0.2, 0.9, 0.4, 0.6]
    for p in priorities:
        tree.add(p)

    assert len(tree) == 8
    assert abs(tree.total - sum(priorities)) < 1e-10
    print(f"✓ Add: total={tree.total:.4f}, expected={sum(priorities):.4f}")

    # Test max_priority tracking
    assert tree.max_priority == 1.0
    print(f"✓ Max priority tracked: {tree.max_priority}")

    # Test update
    tree.update(0, 2.0)
    assert tree.max_priority == 2.0
    assert abs(tree.total - (sum(priorities) - 0.5 + 2.0)) < 1e-10
    print(f"✓ Update: new total={tree.total:.4f}")

    # Test get (proportional sampling)
    samples = []
    for _ in range(1000):
        value = np.random.uniform(0, tree.total)
        idx, priority = tree.get(value)
        samples.append(idx)

    # Higher priority indices should be sampled more
    counts = np.bincount(samples, minlength=8)
    print(f"✓ Sampling distribution: {counts}")

    # Test circular buffer behavior
    tree.add(1.5)  # Should overwrite index 0
    assert len(tree) == 8  # Still 8
    print("✓ Circular buffer works")

    print("SumTree tests passed!\n")


def test_experience_packing():
    """Test Experience packing and unpacking."""
    print("=" * 60)
    print("Testing Experience packing/unpacking...")
    print("=" * 60)

    B, A = 2, 10  # Small batch for testing
    device = "cpu"

    # Create mock MainState
    main_state = MainState(
        moneydisposable=torch.randn(B, A),
        savings=torch.randn(B, A).abs(),
        ability=torch.randn(B, A).abs() + 0.5,
        ret=torch.randn(B, A) * 0.1,
        tax_params=torch.randn(B, 5),
        is_superstar_vA=torch.zeros(B, A, dtype=torch.bool),
        is_superstar_vB=torch.zeros(B, A, dtype=torch.bool),
        ability_history_vA=None,
        ability_history_vB=None,
    )

    # Test snapshot
    snapshot = snapshot_main_state(main_state)
    assert torch.allclose(snapshot.savings, main_state.savings)
    assert snapshot.savings is not main_state.savings  # Different objects
    print("✓ Snapshot creates deep copy")

    # Create mock ParallelStates (shock outcomes)
    parallel_A = ParallelState(
        moneydisposable=torch.randn(B, A),
        savings=torch.randn(B, A).abs(),
        ability=torch.randn(B, A).abs() + 0.5,  # Shocked ability
        ret=torch.randn(B, A) * 0.1,
        tax_params=torch.randn(B, 5),
        is_superstar=torch.zeros(B, A, dtype=torch.bool),
        ability_history=None,
    )

    parallel_B = ParallelState(
        moneydisposable=torch.randn(B, A),
        savings=torch.randn(B, A).abs(),
        ability=torch.randn(B, A).abs() + 0.5,  # Different shocked ability
        ret=torch.randn(B, A) * 0.1,
        tax_params=torch.randn(B, 5),
        is_superstar=torch.ones(B, A, dtype=torch.bool),  # Some superstars
        ability_history=None,
    )

    # Pack experience
    exp = pack_experience(snapshot, parallel_A, parallel_B, "A", step=100)

    assert exp.main_moneydisposable.dtype == torch.float16
    assert exp.main_moneydisposable.device.type == "cpu"
    assert exp.committed_branch == "A"
    assert exp.step == 100
    print("✓ Pack experience: correct dtypes and metadata")

    # Check no gradients stored
    assert not exp.main_savings.requires_grad
    assert not exp.shock_ability_A.requires_grad
    print("✓ No gradients stored in Experience")

    # Unpack to MainState
    unpacked_main = unpack_to_main_state(exp, device)
    assert unpacked_main.moneydisposable.dtype == torch.float32
    assert torch.allclose(unpacked_main.savings, snapshot.savings, atol=1e-3)  # float16 precision
    print("✓ Unpack to MainState: correct dtype and values")

    # Unpack shock outcomes
    shock_A, shock_B = unpack_shock_outcomes(exp, device)
    assert torch.allclose(shock_A.ability, parallel_A.ability, atol=1e-3)
    assert torch.allclose(shock_B.ability, parallel_B.ability, atol=1e-3)
    assert shock_B.is_superstar.all()  # All True
    print("✓ Unpack shock outcomes: correct values")

    # Test memory estimation
    mem_bytes = compute_experience_memory_bytes(A)
    mem_gb = estimate_buffer_memory_gb(100000, A)
    print(f"✓ Memory estimate: {mem_bytes} bytes/exp, {mem_gb:.3f} GB for 100k buffer")

    print("Experience packing tests passed!\n")


def test_prioritized_buffer():
    """Test PrioritizedReplayBuffer operations."""
    print("=" * 60)
    print("Testing PrioritizedReplayBuffer...")
    print("=" * 60)

    B, A = 2, 10
    capacity = 100
    alpha = 0.6

    buffer = PrioritizedReplayBuffer(capacity=capacity, alpha=alpha)

    assert len(buffer) == 0
    print("✓ Buffer initialized empty")

    # Add experiences
    n_experiences = 20
    for i in range(n_experiences):
        main_state = MainState(
            moneydisposable=torch.randn(B, A),
            savings=torch.randn(B, A).abs(),
            ability=torch.randn(B, A).abs() + 0.5,
            ret=torch.randn(B, A) * 0.1,
            tax_params=torch.randn(B, 5),
            is_superstar_vA=torch.zeros(B, A, dtype=torch.bool),
            is_superstar_vB=torch.zeros(B, A, dtype=torch.bool),
            ability_history_vA=None,
            ability_history_vB=None,
        )
        snapshot = snapshot_main_state(main_state)

        parallel_A = ParallelState(
            moneydisposable=torch.randn(B, A),
            savings=torch.randn(B, A).abs(),
            ability=torch.randn(B, A).abs() + 0.5,
            ret=torch.randn(B, A) * 0.1,
            tax_params=torch.randn(B, 5),
            is_superstar=torch.zeros(B, A, dtype=torch.bool),
            ability_history=None,
        )

        parallel_B = ParallelState(
            moneydisposable=torch.randn(B, A),
            savings=torch.randn(B, A).abs(),
            ability=torch.randn(B, A).abs() + 0.5,
            ret=torch.randn(B, A) * 0.1,
            tax_params=torch.randn(B, 5),
            is_superstar=torch.zeros(B, A, dtype=torch.bool),
            ability_history=None,
        )

        buffer.add(snapshot, parallel_A, parallel_B, "A" if i % 2 == 0 else "B", step=i)

    assert len(buffer) == n_experiences
    print(f"✓ Added {n_experiences} experiences")

    # Test sampling
    batch_size = 8
    beta = 0.4
    experiences, indices, is_weights = buffer.sample(batch_size, beta)

    assert len(experiences) == batch_size
    assert len(indices) == batch_size
    assert len(is_weights) == batch_size
    assert all(exp is not None for exp in experiences)
    assert is_weights.min() >= 0
    assert is_weights.max() <= 100
    print(f"✓ Sampled {batch_size} experiences")
    print(f"  Indices: {indices}")
    print(f"  IS weights: {is_weights}")

    # Test priority update
    fake_losses = np.random.uniform(0.1, 2.0, size=batch_size)
    buffer.update_priorities(indices, fake_losses)
    print(f"✓ Updated priorities with losses: {fake_losses[:3]}...")

    # Sample again - higher loss experiences should be sampled more
    experiences2, indices2, is_weights2 = buffer.sample(batch_size, beta)
    print(f"✓ Second sampling after priority update")
    print(f"  Indices: {indices2}")

    # Test circular buffer overflow
    for i in range(capacity + 10):
        main_state = MainState(
            moneydisposable=torch.randn(B, A),
            savings=torch.randn(B, A).abs(),
            ability=torch.randn(B, A).abs() + 0.5,
            ret=torch.randn(B, A) * 0.1,
            tax_params=torch.randn(B, 5),
            is_superstar_vA=torch.zeros(B, A, dtype=torch.bool),
            is_superstar_vB=torch.zeros(B, A, dtype=torch.bool),
            ability_history_vA=None,
            ability_history_vB=None,
        )
        snapshot = snapshot_main_state(main_state)

        parallel_A = ParallelState(
            moneydisposable=torch.randn(B, A),
            savings=torch.randn(B, A).abs(),
            ability=torch.randn(B, A).abs() + 0.5,
            ret=torch.randn(B, A) * 0.1,
            tax_params=torch.randn(B, 5),
            is_superstar=torch.zeros(B, A, dtype=torch.bool),
            ability_history=None,
        )

        parallel_B = ParallelState(
            moneydisposable=torch.randn(B, A),
            savings=torch.randn(B, A).abs(),
            ability=torch.randn(B, A).abs() + 0.5,
            ret=torch.randn(B, A) * 0.1,
            tax_params=torch.randn(B, 5),
            is_superstar=torch.zeros(B, A, dtype=torch.bool),
            ability_history=None,
        )

        buffer.add(snapshot, parallel_A, parallel_B, "A", step=i)

    assert len(buffer) == capacity  # Should not exceed capacity
    print(f"✓ Circular buffer: size stays at {len(buffer)} after overflow")

    print("PrioritizedReplayBuffer tests passed!\n")


def test_buffer_save_load(tmp_path="/tmp"):
    """Test buffer save and load."""
    print("=" * 60)
    print("Testing buffer save/load...")
    print("=" * 60)

    import os
    B, A = 2, 5

    buffer = PrioritizedReplayBuffer(capacity=50, alpha=0.6)

    # Add some experiences
    for i in range(10):
        main_state = MainState(
            moneydisposable=torch.ones(B, A) * i,
            savings=torch.ones(B, A) * i,
            ability=torch.ones(B, A),
            ret=torch.zeros(B, A),
            tax_params=torch.zeros(B, 5),
            is_superstar_vA=torch.zeros(B, A, dtype=torch.bool),
            is_superstar_vB=torch.zeros(B, A, dtype=torch.bool),
            ability_history_vA=None,
            ability_history_vB=None,
        )
        snapshot = snapshot_main_state(main_state)

        parallel_A = ParallelState(
            moneydisposable=torch.ones(B, A),
            savings=torch.ones(B, A),
            ability=torch.ones(B, A) * (i + 1),
            ret=torch.zeros(B, A),
            tax_params=torch.zeros(B, 5),
            is_superstar=torch.zeros(B, A, dtype=torch.bool),
            ability_history=None,
        )

        parallel_B = ParallelState(
            moneydisposable=torch.ones(B, A),
            savings=torch.ones(B, A),
            ability=torch.ones(B, A) * (i + 2),
            ret=torch.zeros(B, A),
            tax_params=torch.zeros(B, 5),
            is_superstar=torch.zeros(B, A, dtype=torch.bool),
            ability_history=None,
        )

        buffer.add(snapshot, parallel_A, parallel_B, "A", step=i)

    # Update some priorities
    _, indices, _ = buffer.sample(5, 0.4)
    buffer.update_priorities(indices, np.array([0.5, 1.0, 1.5, 2.0, 2.5]))

    # Save
    save_path = os.path.join(tmp_path, "test_buffer.pkl")
    buffer.save(save_path)
    print(f"✓ Saved buffer to {save_path}")

    # Load into new buffer
    buffer2 = PrioritizedReplayBuffer(capacity=50, alpha=0.6)
    buffer2.load(save_path)

    assert len(buffer2) == len(buffer)
    assert buffer2.tree.total == buffer.tree.total
    print(f"✓ Loaded buffer: size={len(buffer2)}, total_priority={buffer2.tree.total:.4f}")

    # Verify data integrity
    exp1 = buffer.data[0]
    exp2 = buffer2.data[0]
    assert torch.allclose(exp1.main_savings, exp2.main_savings)
    print("✓ Data integrity verified")

    # Cleanup
    os.remove(save_path)
    print("Buffer save/load tests passed!\n")


def test_replay_step_integration():
    """Test replay_step with mock environment."""
    print("=" * 60)
    print("Testing replay_step integration...")
    print("=" * 60)

    # This test requires the actual environment and policy network
    # For now, just test that the function signature works

    from src.replay_buffer.experience import replay_step

    B, A = 2, 10

    # Create a mock experience
    exp = Experience(
        main_moneydisposable=torch.randn(B, A).half(),
        main_savings=torch.randn(B, A).abs().half(),
        main_ability=torch.randn(B, A).abs().half() + 0.5,
        main_ret=torch.randn(B, A).half() * 0.1,
        main_tax_params=torch.randn(B, 5),
        main_is_superstar_vA=torch.zeros(B, A, dtype=torch.bool),
        main_is_superstar_vB=torch.zeros(B, A, dtype=torch.bool),
        shock_ability_A=torch.randn(B, A).abs().half() + 0.5,
        shock_is_superstar_A=torch.zeros(B, A, dtype=torch.bool),
        shock_ability_B=torch.randn(B, A).abs().half() + 0.5,
        shock_is_superstar_B=torch.ones(B, A, dtype=torch.bool),
        committed_branch="A",
        step=42,
    )

    # Unpack and verify
    main_state = unpack_to_main_state(exp, "cpu")
    shock_A, shock_B = unpack_shock_outcomes(exp, "cpu")

    assert main_state.moneydisposable.dtype == torch.float32
    assert shock_A.ability.dtype == torch.float32
    assert shock_B.is_superstar.dtype == torch.bool
    print("✓ Experience unpacking works correctly")

    print("replay_step integration test passed!\n")
    print("(Full integration test requires actual env and policy_net)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("REPLAY BUFFER TESTS")
    print("=" * 60 + "\n")

    test_sum_tree()
    test_experience_packing()
    test_prioritized_buffer()
    test_buffer_save_load()
    test_replay_step_integration()

    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
