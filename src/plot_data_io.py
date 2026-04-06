"""Save and load numerical plot data as .npz files for cross-run comparison."""

import os
import glob
import numpy as np
from typing import Dict, Optional, List, Any


def save_grid_data(
    result_dict: Dict[str, Any],
    plot_id: str,
    step: int,
    exp_name: str,
    save_dir: str
) -> str:
    """
    Save evaluate_on_grid() result dict as .npz file.

    Args:
        result_dict: Return value from PolicyEvaluator.evaluate_on_grid()
        plot_id: Plot identifier, e.g. "A1", "B1", "A1-1h"
        step: Training step
        exp_name: Experiment name
        save_dir: Directory for grid .npz files

    Returns:
        Path to saved .npz file
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{plot_id}_step_{step}.npz")

    save_dict = {
        "x_values": result_dict["x_values"],
        "x_var": np.array(result_dict["x_var"], dtype="U64"),
        "y_var": np.array(result_dict["y_var"], dtype="U64"),
        "color_var": np.array(str(result_dict.get("color_var", "none")), dtype="U64"),
        "color_levels": np.array(result_dict["color_levels"], dtype="U64"),
        "color_values": np.array(result_dict["color_values"], dtype="f8"),
        "exp_name": np.array(exp_name, dtype="U128"),
        "step": np.array(step, dtype="i8"),
        "plot_id": np.array(plot_id, dtype="U64"),
    }

    # Flatten y_values dict: y_{level} -> array
    for level, arr in result_dict["y_values"].items():
        save_dict[f"y_{level}"] = arr

    # Save fixed_values as parallel key/value arrays
    fixed = result_dict.get("fixed_values", {})
    if fixed:
        keys = []
        vals = []
        for k, v in fixed.items():
            keys.append(k)
            vals.append(float(v) if not isinstance(v, str) else float("nan"))
        save_dict["fixed_keys"] = np.array(keys, dtype="U64")
        save_dict["fixed_vals"] = np.array(vals, dtype="f8")

    # Save losses if present
    losses = result_dict.get("losses")
    if losses:
        for loss_name, level_dict in losses.items():
            for level, arr in level_dict.items():
                save_dict[f"loss_{loss_name}_{level}"] = arr

    np.savez(path, **save_dict)
    return path


def load_grid_data(npz_path: str) -> Dict[str, Any]:
    """
    Load .npz file and reconstruct the evaluate_on_grid() result dict.

    Returns:
        Dict with same structure as evaluate_on_grid() output, plus "exp_name", "step", "plot_id".
    """
    data = np.load(npz_path, allow_pickle=False)

    color_levels = list(data["color_levels"])
    color_values = list(data["color_values"])

    # Reconstruct y_values
    y_values = {}
    for level in color_levels:
        key = f"y_{level}"
        if key in data:
            y_values[level] = data[key]

    # Reconstruct fixed_values
    fixed_values = {}
    if "fixed_keys" in data:
        for k, v in zip(data["fixed_keys"], data["fixed_vals"]):
            fixed_values[str(k)] = float(v)

    # Reconstruct losses
    losses = {}
    loss_prefixes = {"fb_loss", "labor_foc_loss", "aux_loss"}
    for key in data.files:
        if key.startswith("loss_"):
            # key format: loss_{loss_name}_{level}
            rest = key[5:]  # strip "loss_"
            for prefix in loss_prefixes:
                if rest.startswith(prefix + "_"):
                    level = rest[len(prefix) + 1:]
                    losses.setdefault(prefix, {})[level] = data[key]
                    break

    color_var_str = str(data["color_var"])
    result = {
        "x_values": data["x_values"],
        "y_values": y_values,
        "color_var": None if color_var_str == "none" else color_var_str,
        "color_levels": color_levels,
        "color_values": color_values,
        "fixed_values": fixed_values,
        "x_var": str(data["x_var"]),
        "y_var": str(data["y_var"]),
        "exp_name": str(data["exp_name"]),
        "step": int(data["step"]),
        "plot_id": str(data["plot_id"]),
    }

    if losses:
        result["losses"] = losses

    return result


def save_input_output_data(
    own_money: np.ndarray,
    own_ability: np.ndarray,
    zeta: np.ndarray,
    mu: np.ndarray,
    labor: np.ndarray,
    per_agent_losses: Optional[Dict[str, np.ndarray]],
    normalizer_stats: Dict[str, float],
    step: int,
    exp_name: str,
    save_dir: str,
    utility: Optional[np.ndarray] = None,
) -> str:
    """
    Save input-output pairwise data as .npz file.

    Args:
        own_money: (N,) normalized own money
        own_ability: (N,) normalized own ability
        zeta: (N,) savings ratio output
        mu: (N,) multiplier output
        labor: (N,) labor output
        per_agent_losses: Dict with keys like "FB", "Euler", "Labor FOC"
        normalizer_stats: Dict with money_mean, money_std, ability_mean, ability_std
        step: Training step
        exp_name: Experiment name
        save_dir: Directory for input_output .npz files

    Returns:
        Path to saved .npz file
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"pairwise_step_{step}.npz")

    save_dict = {
        "own_money": own_money,
        "own_ability": own_ability,
        "zeta": zeta,
        "mu": mu,
        "labor": labor,
        "exp_name": np.array(exp_name, dtype="U128"),
        "step": np.array(step, dtype="i8"),
    }

    # Normalizer stats
    for k, v in normalizer_stats.items():
        save_dict[f"norm_{k}"] = np.array(v, dtype="f8")

    # Per-agent utility
    if utility is not None:
        save_dict["utility"] = utility

    # Per-agent losses
    if per_agent_losses:
        for loss_name, arr in per_agent_losses.items():
            safe_name = loss_name.replace(" ", "_").lower()
            save_dict[f"loss_{safe_name}"] = arr

    np.savez(path, **save_dict)
    return path


def load_input_output_data(npz_path: str) -> Dict[str, Any]:
    """Load input-output pairwise data from .npz file."""
    data = np.load(npz_path, allow_pickle=False)

    result = {
        "own_money": data["own_money"],
        "own_ability": data["own_ability"],
        "zeta": data["zeta"],
        "mu": data["mu"],
        "labor": data["labor"],
        "exp_name": str(data["exp_name"]),
        "step": int(data["step"]),
    }

    # Per-agent utility
    if "utility" in data.files:
        result["utility"] = data["utility"]
    else:
        result["utility"] = None

    # Normalizer stats
    norm_stats = {}
    for key in data.files:
        if key.startswith("norm_"):
            norm_stats[key[5:]] = float(data[key])
    result["normalizer_stats"] = norm_stats

    # Per-agent losses
    losses = {}
    for key in data.files:
        if key.startswith("loss_"):
            losses[key[5:]] = data[key]
    result["per_agent_losses"] = losses

    return result


def save_run_meta(config, save_dir: str) -> str:
    """
    Save lightweight config summary as meta.npz.

    Args:
        config: SimpleNamespace config object
        save_dir: plot_data directory
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "meta.npz")

    save_dict = {
        "exp_name": np.array(getattr(config, "exp_name", "unknown"), dtype="U128"),
        "n_agents": np.array(config.training.agents, dtype="i8"),
        "save_interval": np.array(config.training.save_interval, dtype="i8"),
        "training_steps": np.array(config.training.training_steps, dtype="i8"),
        "sigma_v": np.array(config.shock.sigma_v, dtype="f8"),
        "rho_v": np.array(config.shock.rho_v, dtype="f8"),
    }

    # Optional fields
    if hasattr(config, "bewley_model"):
        save_dict["fix"] = np.array(getattr(config.bewley_model, "fix", False))
        for attr in ["theta", "beta", "r", "gamma"]:
            if hasattr(config.bewley_model, attr):
                save_dict[attr] = np.array(getattr(config.bewley_model, attr), dtype="f8")

    np.savez(path, **save_dict)
    return path


def load_run_meta(meta_path: str) -> Dict[str, Any]:
    """Load config summary from meta.npz."""
    data = np.load(meta_path, allow_pickle=False)
    result = {}
    for key in data.files:
        val = data[key]
        if val.dtype.kind == "U":
            result[key] = str(val)
        elif val.dtype.kind == "b":
            result[key] = bool(val)
        elif val.dtype.kind == "i":
            result[key] = int(val)
        else:
            result[key] = float(val)
    return result


def discover_runs(checkpoints_dir: str) -> List[Dict[str, Any]]:
    """
    Find all runs with plot_data/ directories.

    Returns:
        List of dicts: [{"exp_name": str, "path": str, "meta": dict,
                         "grid_steps": [int], "grid_plot_ids": [str],
                         "io_steps": [int]}, ...]
    """
    runs = []
    # Look for plot_data directories inside checkpoint subdirs
    for entry in sorted(os.listdir(checkpoints_dir)):
        plot_data_dir = os.path.join(checkpoints_dir, entry, "plot_data")
        if not os.path.isdir(plot_data_dir):
            continue

        run_info = {"exp_name": entry, "path": plot_data_dir}

        # Load meta if available
        meta_path = os.path.join(plot_data_dir, "meta.npz")
        if os.path.exists(meta_path):
            run_info["meta"] = load_run_meta(meta_path)
        else:
            run_info["meta"] = {"exp_name": entry}

        # Discover grid data
        grid_dir = os.path.join(plot_data_dir, "grid")
        grid_steps = set()
        grid_plot_ids = set()
        if os.path.isdir(grid_dir):
            for f in os.listdir(grid_dir):
                if f.endswith(".npz"):
                    # Format: {plot_id}_step_{step}.npz
                    parts = f[:-4].rsplit("_step_", 1)
                    if len(parts) == 2:
                        grid_plot_ids.add(parts[0])
                        grid_steps.add(int(parts[1]))
        run_info["grid_steps"] = sorted(grid_steps)
        run_info["grid_plot_ids"] = sorted(grid_plot_ids)

        # Discover input-output data
        io_dir = os.path.join(plot_data_dir, "input_output")
        io_steps = set()
        if os.path.isdir(io_dir):
            for f in os.listdir(io_dir):
                if f.startswith("pairwise_step_") and f.endswith(".npz"):
                    step_str = f[len("pairwise_step_"):-4]
                    io_steps.add(int(step_str))
        run_info["io_steps"] = sorted(io_steps)

        # Discover panel data (cluster tracking)
        panel_dir = os.path.join(plot_data_dir, "panel")
        panel_steps = set()
        if os.path.isdir(panel_dir):
            for f in os.listdir(panel_dir):
                if f.startswith("panel_step_") and f.endswith(".npz"):
                    step_str = f[len("panel_step_"):-4]
                    try:
                        panel_steps.add(int(step_str))
                    except ValueError:
                        pass
        run_info["panel_steps"] = sorted(panel_steps)

        runs.append(run_info)

    return runs


# ---------------------------------------------------------------------------
# Panel data save / load (for cluster tracking)
# ---------------------------------------------------------------------------

def save_panel_data(panel_buffer, step: int, exp_name: str, save_dir: str) -> str:
    """
    Save a PanelBuffer to a .npz file.

    Args:
        panel_buffer: PanelBuffer instance.
        step: Training step.
        exp_name: Experiment name (stored in metadata).
        save_dir: Directory for panel .npz files (plot_data/panel/).

    Returns:
        Path to saved .npz file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"panel_step_{step}.npz")
    panel_buffer.to_npz(path)
    return path


def load_panel_data(npz_path: str):
    """
    Load a PanelBuffer from a .npz file.

    Args:
        npz_path: Path to panel .npz file.

    Returns:
        PanelBuffer instance.
    """
    from src.cluster_analysis import PanelBuffer
    return PanelBuffer.from_npz(npz_path)
