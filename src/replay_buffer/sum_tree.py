# src/replay_buffer/sum_tree.py
"""
SumTree data structure for O(log N) proportional sampling.

Implementation follows the standard sum tree for Prioritized Experience Replay.
The tree is stored as a flat array where:
- tree[0] is the root (sum of all priorities)
- For node i: left child = 2*i+1, right child = 2*i+2, parent = (i-1)//2
- Leaf nodes start at index (capacity - 1)
"""

import numpy as np
from typing import Tuple


class SumTree:
    """
    Array-based sum tree for O(log N) proportional sampling.

    Properties:
        - add(): O(log N) - add new priority
        - update(): O(log N) - update existing priority
        - get(): O(log N) - sample proportional to priority
        - total: O(1) - sum of all priorities (root node)
    """

    def __init__(self, capacity: int):
        """
        Initialize SumTree.

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        # Tree array: 2*capacity - 1 nodes total
        # Leaf nodes: indices [capacity-1, 2*capacity-2]
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0  # Current write position (circular)
        self.n_entries = 0  # Number of entries currently stored
        self.max_priority = 1.0  # Track max priority for new experiences (Line 2: p_1 = 1)

    def _propagate(self, idx: int, change: float) -> None:
        """
        Propagate priority change from leaf up to root.

        Args:
            idx: Leaf index in tree array
            change: Priority change (new - old)
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _leaf_idx(self, data_idx: int) -> int:
        """Convert data index to tree leaf index."""
        return data_idx + self.capacity - 1

    def add(self, priority: float) -> int:
        """
        Add new entry with given priority.

        Args:
            priority: Priority value for the new entry

        Returns:
            data_idx: Index in the data buffer where this entry is stored
        """
        data_idx = self.data_pointer
        tree_idx = self._leaf_idx(data_idx)

        # Update tree
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

        # Update max priority
        if priority > self.max_priority:
            self.max_priority = priority

        # Move pointer (circular buffer)
        self.data_pointer = (self.data_pointer + 1) % self.capacity

        # Track number of entries
        if self.n_entries < self.capacity:
            self.n_entries += 1

        return data_idx

    def update(self, data_idx: int, priority: float) -> None:
        """
        Update priority at given data index.

        Args:
            data_idx: Index in data buffer
            priority: New priority value
        """
        tree_idx = self._leaf_idx(data_idx)
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

        # Update max priority if new priority is higher
        if priority > self.max_priority:
            self.max_priority = priority

    def get(self, value: float) -> Tuple[int, float]:
        """
        Sample leaf index proportional to priority.

        Given a value in [0, total], traverse tree to find the leaf
        whose cumulative priority range contains this value.

        Args:
            value: Random value in [0, total]

        Returns:
            data_idx: Index in data buffer
            priority: Priority value at that index
        """
        idx = 0  # Start at root

        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2

            # If we've reached a leaf
            if left >= len(self.tree):
                break

            # Choose left or right based on cumulative sum
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right

        data_idx = idx - (self.capacity - 1)
        priority = self.tree[idx]

        return data_idx, priority

    @property
    def total(self) -> float:
        """Return sum of all priorities (root node)."""
        return self.tree[0]

    @property
    def min_priority(self) -> float:
        """
        Return minimum priority among stored entries.
        Used for IS weight normalization.
        """
        # Only look at leaf nodes that have data
        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.n_entries
        if self.n_entries == 0:
            return 0.0

        # Get minimum non-zero priority
        priorities = self.tree[leaf_start:leaf_end]
        non_zero = priorities[priorities > 0]
        if len(non_zero) == 0:
            return 0.0
        return float(np.min(non_zero))

    def __len__(self) -> int:
        """Return number of entries stored."""
        return self.n_entries
