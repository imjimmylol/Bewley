"""
FocalAgentTracker — tracks selected agents across every training step.

Selection strategy: ability quintiles × savings coverage at t=0.
This lets you compare agents with similar ability but different savings history,
diagnosing why same-ability agents diverge in decision rule outcomes.

Storage: pre-allocated float32 numpy array (max_steps, n_focal, N_VARS).
Saved as .npz (compressed) + _meta.pkl.

Variable layout (VAR_NAMES order):
  0: a_t    — initial savings entering period t  (main_state.savings BEFORE env.step)
  1: m_t    — disposable money at t
  2: v_t    — ability at t
  3: s_t    — superstar flag at t (float 0/1)
  4: c_t    — consumption decision at t
  5: a_tp1  — savings decision at t (= next period's assets)
  6: y_t    — income before tax at t
  7: mu_t   — borrowing constraint multiplier at t
  8: income_tax — income tax at t
  9: savings_tax — savings tax at t
  10: wage — wage at t
  11: ret — return on savings at t
  12: l_t — labor choice at t
"""

import os
import pickle
import numpy as np

VAR_NAMES = ["a_t", "m_t", "v_t", "s_t", "c_t", "a_tp1", "y_t", "mu_t",
             "income_tax", "savings_tax", "wage", "ret", "l_t"]
_N_VARS = len(VAR_NAMES)

# Colors for ability groups in overlays (up to 8 groups)
GROUP_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3",
                "#a65628", "#f781bf", "#999999"]


class FocalAgentTracker:

    def __init__(
        self,
        focal_indices: np.ndarray,   # (N_focal, 2): (batch_idx, agent_idx)
        focal_metadata: dict,
        max_steps: int,
    ):
        self.focal_indices = focal_indices
        self.focal_metadata = focal_metadata
        self.n_focal = len(focal_indices)

        self._data = np.full((max_steps, self.n_focal, _N_VARS), np.nan, dtype=np.float32)
        self._steps = np.zeros(max_steps, dtype=np.int32)
        self._cursor = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def initialize(
        cls,
        main_state,
        max_steps: int,
        n_ability_groups: int = 5,
        n_per_group: int = 10,
    ) -> "FocalAgentTracker":
        """
        Select focal agents from the initial main_state.

        Within each ability quintile, picks n_per_group agents that span
        the savings distribution as evenly as possible.
        Agents from all batches are eligible.
        """
        ability = main_state.ability.detach().cpu().numpy()   # (B, A)
        savings = main_state.savings.detach().cpu().numpy()   # (B, A)
        B, A = ability.shape

        batch_ids = np.repeat(np.arange(B), A)
        agent_ids = np.tile(np.arange(A), B)
        ability_flat = ability.flatten()
        savings_flat = savings.flatten()

        ability_edges = np.percentile(
            ability_flat, np.linspace(0, 100, n_ability_groups + 1)
        )

        selected_flat = []   # flat indices into (B*A,)
        group_info = []

        for g in range(n_ability_groups):
            lo, hi = ability_edges[g], ability_edges[g + 1]
            if g < n_ability_groups - 1:
                in_group = np.where((ability_flat >= lo) & (ability_flat < hi))[0]
            else:
                in_group = np.where((ability_flat >= lo) & (ability_flat <= hi))[0]

            if len(in_group) == 0:
                continue

            sav_in_group = savings_flat[in_group]
            target_pcts = np.linspace(0, 100, n_per_group)
            target_savings = np.percentile(sav_in_group, target_pcts)

            seen = set()
            group_flat = []
            for ts in target_savings:
                flat_idx = in_group[np.argmin(np.abs(savings_flat[in_group] - ts))]
                if flat_idx not in seen:
                    seen.add(flat_idx)
                    group_flat.append(int(flat_idx))

            selected_flat.extend(group_flat)

            group_info.append({
                "ability_range": (float(lo), float(hi)),
                "ability_mean": float(ability_flat[in_group].mean()),
                "ability_std":  float(ability_flat[in_group].std()),
                "n_selected":   len(group_flat),
                "init_savings": [float(savings_flat[i]) for i in group_flat],
            })

        focal_indices = np.array(
            [(int(batch_ids[i]), int(agent_ids[i])) for i in selected_flat],
            dtype=np.int32,
        )

        # Build group_id array: which ability group each focal agent belongs to
        group_ids = np.zeros(len(selected_flat), dtype=np.int32)
        cursor = 0
        for g_idx, g in enumerate(group_info):
            group_ids[cursor: cursor + g["n_selected"]] = g_idx
            cursor += g["n_selected"]

        metadata = {
            "var_names": VAR_NAMES,
            "groups": group_info,
            "group_ids": group_ids.tolist(),
            "n_ability_groups": n_ability_groups,
            "n_per_group": n_per_group,
            "total_focal": len(focal_indices),
        }

        print(
            f"[FocalAgentTracker] Selected {len(focal_indices)} focal agents "
            f"({n_ability_groups} ability groups × up to {n_per_group} savings levels)"
        )
        for i, g in enumerate(group_info):
            lo, hi = g["ability_range"]
            print(f"  group {i}: ability [{lo:.2f}, {hi:.2f}]  "
                  f"mean={g['ability_mean']:.2f}  {g['n_selected']} agents")

        return cls(focal_indices, metadata, max_steps)

    # ------------------------------------------------------------------
    # Per-step recording
    # ------------------------------------------------------------------

    def record(self, step: int, a_t_pre: np.ndarray, temp_state) -> None:
        """
        Record focal agents at this training step.

        Args:
            step:      current training step number
            a_t_pre:   (B, A) float32 numpy array — main_state.savings captured
                       BEFORE env.step(), i.e. the savings entering period t.
            temp_state: TemporaryState from env.step()
        """
        if self._cursor >= len(self._data):
            return

        B, A = a_t_pre.shape

        def _np(tensor):
            return tensor.detach().cpu().numpy()

        buf = np.empty((B, A, _N_VARS), dtype=np.float32)
        buf[..., 0] = a_t_pre
        buf[..., 1] = _np(temp_state.money_disposable)
        buf[..., 2] = _np(temp_state.ability)
        buf[..., 3] = _np(temp_state.is_superstar_vA).astype(np.float32)
        buf[..., 4] = _np(temp_state.consumption)
        buf[..., 5] = _np(temp_state.savings)
        buf[..., 6] = _np(temp_state.income_before_tax)
        buf[..., 7] = (
            _np(temp_state.mu).astype(np.float32)
            if temp_state.mu is not None
            else np.zeros((B, A), dtype=np.float32)
        )
        buf[..., 8]  = _np(temp_state.income_tax)
        buf[..., 9]  = _np(temp_state.savings_tax)
        buf[..., 10] = _np(temp_state.wage)
        buf[..., 11] = _np(temp_state.ret)
        buf[..., 12] = _np(temp_state.labor)

        b_idx = self.focal_indices[:, 0]
        a_idx = self.focal_indices[:, 1]
        self._data[self._cursor] = buf[b_idx, a_idx]
        self._steps[self._cursor] = step
        self._cursor += 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        valid_data = self._data[:self._cursor]
        valid_steps = self._steps[:self._cursor]

        np.savez_compressed(
            path,
            data=valid_data,
            steps=valid_steps,
            focal_indices=self.focal_indices,
        )
        meta_path = path.replace(".npz", "_meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(self.focal_metadata, f)

        size_mb = valid_data.nbytes / 1e6
        print(f"[FocalAgentTracker] Saved {self._cursor} steps × {self.n_focal} agents "
              f"→ {path} ({size_mb:.1f} MB uncompressed)")

    @classmethod
    def load(cls, path: str) -> "FocalAgentTracker":
        arr = np.load(path)
        data = arr["data"]
        steps = arr["steps"]
        focal_indices = arr["focal_indices"]

        meta_path = path.replace(".npz", "_meta.pkl")
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        n_steps, n_focal_stored, n_vars_stored = data.shape
        tracker = cls(focal_indices, metadata, max_steps=n_steps)
        # Backward-compatible: old files may have fewer columns; new cols stay NaN
        tracker._data[:n_steps, :, :n_vars_stored] = data
        tracker._steps[:n_steps] = steps
        tracker._cursor = n_steps
        return tracker

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def get_trajectory(self, focal_idx: int) -> dict:
        """Full time-series for one focal agent."""
        valid = self._data[:self._cursor, focal_idx, :]
        return {
            "steps":     self._steps[:self._cursor].copy(),
            "batch_idx": int(self.focal_indices[focal_idx, 0]),
            "agent_idx": int(self.focal_indices[focal_idx, 1]),
            "group_id":  self.focal_metadata["group_ids"][focal_idx],
            **{name: valid[:, i].copy() for i, name in enumerate(VAR_NAMES)},
        }

    def to_dataframe(self):
        """All trajectories as a pandas DataFrame (long format)."""
        import pandas as pd
        T = self._cursor
        group_ids = self.focal_metadata["group_ids"]
        frames = []
        for f in range(self.n_focal):
            b, a = self.focal_indices[f]
            chunk = pd.DataFrame(self._data[:T, f, :], columns=VAR_NAMES)
            chunk["step"] = self._steps[:T]
            chunk["focal_idx"] = f
            chunk["batch_idx"] = int(b)
            chunk["agent_idx"] = int(a)
            chunk["group_id"] = group_ids[f]
            frames.append(chunk)
        return pd.concat(frames, ignore_index=True)

    def current_plot_overlay(self, features_tensor, zeta_tensor, mu_tensor, labor_tensor):
        """
        Extract focal agents' current-step positions for overlaying on plots.

        Args:
            features_tensor: (B, A, 2A+2) normalized features tensor (from build_inputs)
            zeta_tensor:     (B, A) savings ratio tensor
            mu_tensor:       (B, A) multiplier tensor
            labor_tensor:    (B, A) labor tensor

        Returns dict with arrays of shape (N_focal,) ready for scatter overlay:
            norm_money, norm_ability, zeta, mu, labor, group_ids, colors
        """
        import torch
        b_idx = self.focal_indices[:, 0]
        a_idx = self.focal_indices[:, 1]

        def _extract(t):
            if torch.is_tensor(t):
                return t[b_idx, a_idx].detach().cpu().numpy()
            return t[b_idx, a_idx]

        group_ids = np.array(self.focal_metadata["group_ids"])
        colors = [GROUP_COLORS[g % len(GROUP_COLORS)] for g in group_ids]
        n_groups = self.focal_metadata["n_ability_groups"]
        group_info = self.focal_metadata["groups"]

        return {
            "norm_money":   _extract(features_tensor[..., -2]),
            "norm_ability": _extract(features_tensor[..., -1]),
            "zeta":         _extract(zeta_tensor),
            "mu":           _extract(mu_tensor),
            "labor":        _extract(labor_tensor),
            "group_ids":    group_ids,
            "colors":       colors,
            "n_groups":     n_groups,
            "group_labels": [
                f"g{g} v∈[{gi['ability_range'][0]:.1f},{gi['ability_range'][1]:.1f}]"
                for g, gi in enumerate(group_info)
            ],
        }
