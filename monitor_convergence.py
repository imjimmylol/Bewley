"""
Monitor convergence of the consumption rule to its ergodic equilibrium.

Loads a *frozen* policy from a checkpoint, re-initialises agents from
scratch (not from the saved state), then simulates forward and plots whether
the cross-sectional c(m_t) policy function stabilises over time.

Usage
-----
python monitor_convergence.py \
    --checkpoint_dir checkpoints/Bewley_Hetero_ability_low_uncer_nonlinear_HardNorm \
    --step 50000 \
    --config config/0406/incomplete_nonlinear_low_uncer_hn.yaml \
    --n_sim_steps 300 \
    --snapshot_every 10 \
    --output convergence.png
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# ── project imports ──────────────────────────────────────────────────────────
from src.train import initialize_env_state
from src.environment import EconomyEnv
from src.normalizer import HardNormalizer, RunningPerAgentWelford
from src.models.model import FiLMResNet2In
from src.utils.configloader import load_configs, dict_to_namespace, compute_derived_params
from src.calloss import LossCalculator
from src.calloss import AuxLossMu

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Checkpoint loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pick_weight_path(weights_dir: str, step: int | None) -> str:
    if step is None:
        p = os.path.join(weights_dir, "model_final.pt")
        if os.path.exists(p):
            return p
        # fall back to the highest numbered step
        candidates = sorted(
            [f for f in os.listdir(weights_dir) if f.startswith("model_step_")],
            key=lambda f: int(f.split("_")[-1].split(".")[0])
        )
        if not candidates:
            raise FileNotFoundError(f"No model weights found in {weights_dir}")
        return os.path.join(weights_dir, candidates[-1])
    return os.path.join(weights_dir, f"model_step_{step}.pt")


def load_checkpoint(checkpoint_dir: str, step: int | None, config, device: str):
    """
    Load policy network + normalizer from a checkpoint directory.

    Parameters
    ----------
    checkpoint_dir : str
        Root directory, e.g. ``checkpoints/Bewley_Hetero_…``.
    step : int or None
        Which snapshot to load.  None → load model_final.pt.
    config : SimpleNamespace
        Parsed config (needed to build the model architecture).
    device : str
    """
    weights_dir    = os.path.join(checkpoint_dir, "weights")
    normalizer_dir = os.path.join(checkpoint_dir, "normalizer")

    # ── model ──────────────────────────────────────────────────────────────
    weight_path = _pick_weight_path(weights_dir, step)
    print(f"  Loading weights : {weight_path}")

    # Infer n_agents from the checkpoint weights so the architecture always matches
    sd = torch.load(weight_path, map_location=device)
    state_dim = sd["state_encoder.0.weight"].shape[1]
    n_agents = (state_dim - 2) // 2
    print(f"  Inferred n_agents from checkpoint: {n_agents}")

    policy_net = FiLMResNet2In(
        state_dim=state_dim,
        cond_dim=5,
        output_dim=3,
    ).to(device)
    policy_net.load_state_dict(sd)
    policy_net.eval()

    # ── normalizer ─────────────────────────────────────────────────────────
    norm_type = getattr(config.training, "normalizer", "welford")
    if norm_type == "hard":
        v_min = getattr(config.initial_state, "v_min", 0)
        v_max = getattr(config.initial_state, "v_max", 100)
        normalizer = HardNormalizer(fixed_bounds={"ability": (v_min, v_max)})
    else:
        normalizer = RunningPerAgentWelford(batch_dim=0, agent_dim=None)

    # Find the matching normalizer snapshot
    if step is not None:
        norm_path = os.path.join(normalizer_dir, f"norm_step_{step}.pt")
    else:
        candidates = sorted(
            [f for f in os.listdir(normalizer_dir) if f.startswith("norm_step_")],
            key=lambda f: int(f.split("_")[-1].split(".")[0])
        )
        norm_path = os.path.join(normalizer_dir, candidates[-1]) if candidates else None

    if norm_path and os.path.exists(norm_path):
        print(f"  Loading normalizer: {norm_path}")
        normalizer.load(norm_path)
    else:
        print("  [warn] No normalizer checkpoint found; normalizer is untrained.")

    return policy_net, normalizer, n_agents


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Cross-sectional binned c(m) — the key summary statistic
# ─────────────────────────────────────────────────────────────────────────────

def _binned_c_m(m: np.ndarray, c: np.ndarray, n_bins: int = 30):
    """
    Bin money-on-hand ``m`` into ``n_bins`` quantile-spaced buckets and return
    (bin_centres, mean_c_per_bin).  Bins with fewer than 3 observations are
    dropped so noisy endpoints don't dominate.
    """
    m, c = m.flatten(), c.flatten()
    edges = np.quantile(m, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)           # collapse identical quantiles
    centres, means, stds = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (m >= lo) & (m < hi)
        if mask.sum() < 3:
            continue
        centres.append(0.5 * (lo + hi))
        means.append(c[mask].mean())
        stds.append(c[mask].std())
    return np.array(centres), np.array(means), np.array(stds)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(env, policy_net, main_state,
                   n_steps: int, snapshot_every: int,
                   fix_ability: bool, config=None):
    """
    Run the economy forward for ``n_steps`` periods with a *frozen* policy.

    Returns
    -------
    snapshots : list of dict
        Each entry (taken every ``snapshot_every`` steps) contains::

            {
                "step":  int,
                "m_t":   np.ndarray (B*A,),
                "c_t":   np.ndarray (B*A,),
                "a_t":   np.ndarray (B*A,),
                "zeta":  np.ndarray (B*A,),
                "wage":  float,   # mean equilibrium wage
                "ret":   float,   # mean equilibrium return
            }
    """
    policy_net.eval()
    snapshots    = []
    total_losses = []

    loss_calc = (
        LossCalculator(config, device=str(next(policy_net.parameters()).device))
        if config is not None else None
    )

    # Pick one focal agent to trace throughout the simulation
    n_batch  = main_state.savings.shape[0]
    n_agents = main_state.savings.shape[1]
    focal_b  = np.random.randint(0, n_batch)
    focal_a  = np.random.randint(0, n_agents)
    print(f"\nRunning {n_steps} simulation steps (snapshot every {snapshot_every})…")
    print(f"  Focal agent: batch={focal_b}, agent={focal_a}")
    print(f"  {'step':>5}  {'zeta_focal':>10}  {'m_focal':>8}  {'wage':>8}  {'ret':>8}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")

    with torch.no_grad():
        for step in range(1, n_steps + 1):
            main_state, temp_state, (pA, oA), (pB, oB) = env.step(
                main_state=main_state,
                policy_net=policy_net,
                deterministic=False,
                fix=fix_ability,
                update_normalizer=False,
                commit_strategy="random",
            )

            if loss_calc is not None:
                losses = loss_calc.compute_all_losses(
                    consumption_t=temp_state.consumption,
                    labor_t=temp_state.labor,
                    ibt=temp_state.income_before_tax,
                    savings_ratio_t=temp_state.savings_ratio,
                    mu_t=temp_state.mu,
                    wage_t=temp_state.wage,
                    ret_t=temp_state.ret,
                    money_disposable_t=temp_state.money_disposable,
                    ability_t=temp_state.ability,
                    consumption_A_tp1=oA["consumption"],
                    consumption_B_tp1=oB["consumption"],
                    ibt_A_tp1=oA["income_before_tax"],
                    ibt_B_tp1=oB["income_before_tax"],
                )
                total_losses.append(losses["total"].item())

            if step % snapshot_every == 0 or step == 1:
                m_np    = temp_state.money_disposable.detach().cpu().numpy().flatten()
                c_np    = temp_state.consumption.detach().cpu().numpy().flatten()
                a_np    = temp_state.savings.detach().cpu().numpy().flatten()
                zeta_np = temp_state.savings_ratio.detach().cpu().numpy().flatten()
                wage_np = float(temp_state.wage.detach().cpu().numpy().mean())
                ret_np  = float(temp_state.ret.detach().cpu().numpy().mean())

                zeta_focal = float(temp_state.savings_ratio[focal_b, focal_a].detach().cpu())
                m_focal    = float(temp_state.money_disposable[focal_b, focal_a].detach().cpu())

                snapshots.append({
                    "step":        step,
                    "m_t":         m_np,
                    "c_t":         c_np,
                    "a_t":         a_np,
                    "zeta":        zeta_np,
                    "wage":        wage_np,
                    "ret":         ret_np,
                    "zeta_focal":  zeta_focal,
                    "m_focal":     m_focal,
                })

                print(f"  {step:5d}  {zeta_focal:10.4f}  {m_focal:8.3f}  {wage_np:8.4f}  {ret_np:8.4f}")

                if step % (snapshot_every * 10) == 0:
                    loss_str = f"  loss={total_losses[-1]:.4f}" if total_losses else ""
                    print(f"         [agg] mean(m)={m_np.mean():.3f}"
                          f"  mean(c)={c_np.mean():.3f}{loss_str}")

            del temp_state, pA, pB, oA, oB

    print(f"Simulation complete.  Collected {len(snapshots)} snapshots.")
    if total_losses:
        final_loss = float(np.mean(total_losses[-max(1, len(total_losses)//10):]))
        print(f"  final loss (last 10% avg):  {final_loss:.6f}")

    return snapshots


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Ergodic means across training checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def detect_training_steps(checkpoint_dir: str) -> list[int]:
    """Return sorted list of training steps that have saved weights."""
    weights_dir = os.path.join(checkpoint_dir, "weights")
    steps = sorted(
        int(f.split("_")[-1].split(".")[0])
        for f in os.listdir(weights_dir)
        if f.startswith("model_step_")
    )
    return steps


def compute_ergodic_means(
    checkpoint_dir: str,
    train_step: int,
    config,
    device: str,
    n_burn: int = 150,
    n_avg: int = 50,
) -> dict:
    """
    Load checkpoint at `train_step`, run `n_burn + n_avg` simulation steps
    from a fresh initial state, and return ergodic means averaged over
    the last `n_avg` steps and all agents.
    """
    policy_net, normalizer, ckpt_n_agents = load_checkpoint(
        checkpoint_dir, train_step, config, device
    )
    if ckpt_n_agents != config.training.agents:
        config.training.agents = ckpt_n_agents

    main_state = initialize_env_state(config, torch.device(device))
    env = EconomyEnv(config, normalizer=normalizer, device=torch.device(device))
    fix_ability = getattr(config.bewley_model, "fix", False)

    totals = {"c": [], "m": [], "a": []}
    policy_net.eval()
    with torch.no_grad():
        for t in range(1, n_burn + n_avg + 1):
            main_state, temp_state, (pA, oA), (pB, oB) = env.step(
                main_state=main_state,
                policy_net=policy_net,
                deterministic=False,
                fix=fix_ability,
                update_normalizer=False,
                commit_strategy="random",
            )
            if t > n_burn:
                totals["c"].append(temp_state.consumption.detach().cpu().numpy().mean())
                totals["m"].append(temp_state.money_disposable.detach().cpu().numpy().mean())
                totals["a"].append(temp_state.savings.detach().cpu().numpy().mean())
            del temp_state, pA, pB, oA, oB

    return {
        "train_step": train_step,
        "mean_c":  float(np.mean(totals["c"])),
        "mean_m":  float(np.mean(totals["m"])),
        "mean_a":  float(np.mean(totals["a"])),
        "std_c":   float(np.std(totals["c"])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Convergence metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_convergence_metric(snapshots, n_bins=30):
    """
    For each snapshot compute the L2 distance of c(m) from the *final* snapshot.

    Returns parallel arrays ``(steps, distances)``.
    """
    # Build c(m) curve for each snapshot on a shared grid
    all_m = np.concatenate([s["m_t"] for s in snapshots])
    grid  = np.quantile(all_m, np.linspace(0.05, 0.95, n_bins))

    def _interp(snap):
        centres, means, _ = _binned_c_m(snap["m_t"], snap["c_t"], n_bins=n_bins)
        if len(centres) < 2:
            return np.full(len(grid), np.nan)
        return np.interp(grid, centres, means)

    curves = [_interp(s) for s in snapshots]
    ref    = curves[-1]

    steps, dists = [], []
    for s, curve in zip(snapshots, curves):
        mask = ~np.isnan(curve) & ~np.isnan(ref)
        if mask.sum() < 2:
            continue
        steps.append(s["step"])
        dists.append(np.sqrt(np.mean((curve[mask] - ref[mask]) ** 2)))
    return np.array(steps), np.array(dists)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(snapshots, save_path: str, checkpoint_label: str,
                     n_bins: int = 30):
    """
    Four-panel convergence figure.

    Panel B  ─  Aggregate statistics over time (mean c, mean m, mean a)
    Panel W  ─  Equilibrium wage over time
    Panel C  ─  L2 convergence metric vs final period
    Panel R  ─  Equilibrium return over time
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Convergence to ergodic equilibrium\n{checkpoint_label}",
        fontsize=13, fontweight="bold"
    )

    steps  = [s["step"] for s in snapshots]
    mean_c = np.array([s["c_t"].mean() for s in snapshots])
    mean_m = np.array([s["m_t"].mean() for s in snapshots])
    mean_a = np.array([s["a_t"].mean() for s in snapshots])
    std_c  = np.array([s["c_t"].std()  for s in snapshots])
    wages  = np.array([s["wage"]        for s in snapshots])
    rets   = np.array([s["ret"]         for s in snapshots])

    # ── Panel B: ergodic means ────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(steps, mean_c, label="mean $c_t$",      color="steelblue",  lw=1.5)
    ax.plot(steps, mean_m, label="mean $m_t$",      color="darkorange", lw=1.5, ls="--")
    ax.plot(steps, mean_a, label="mean $a_{t+1}$",  color="seagreen",   lw=1.5, ls=":")
    ax.fill_between(steps, mean_c - std_c, mean_c + std_c,
                    alpha=0.15, color="steelblue", label="±1 sd($c_t$)")
    ax.set_xlabel("simulation step")
    ax.set_ylabel("mean across agents")
    ax.set_title("B — Aggregate dynamics\n(converging → ergodic distribution)")
    ax.legend(fontsize=8)

    # ── Panel W: equilibrium wage ─────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(steps, wages, color="mediumpurple", lw=1.5)
    ax.axhline(wages[-20:].mean(), color="mediumpurple", ls=":", alpha=0.6,
               label=f"late-mean={wages[-20:].mean():.4f}")
    ax.set_xlabel("simulation step")
    ax.set_ylabel("equilibrium wage  $w_t$")
    ax.set_title("W — Equilibrium wage over simulation")
    ax.legend(fontsize=8)

    # ── Panel C: L2 convergence ───────────────────────────────────────────
    ax = axes[1, 0]
    conv_steps, conv_dists = _compute_convergence_metric(snapshots, n_bins=n_bins)
    ax.semilogy(conv_steps, conv_dists, color="crimson", lw=1.5)
    ax.set_xlabel("simulation step")
    ax.set_ylabel("L2 distance from final $c(m)$  [log scale]")
    ax.set_title("C — Convergence metric  (→ 0 = at equilibrium)")
    ax.axhline(np.min(conv_dists) * 2, color="crimson", ls=":", alpha=0.5,
               label="2×minimum")
    ax.legend(fontsize=8)

    # ── Panel R: equilibrium return ───────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(steps, rets, color="coral", lw=1.5)
    ax.axhline(rets[-20:].mean(), color="coral", ls=":", alpha=0.6,
               label=f"late-mean={rets[-20:].mean():.4f}")
    ax.set_xlabel("simulation step")
    ax.set_ylabel("equilibrium return  $r_t$")
    ax.set_title("R — Equilibrium return over simulation")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"\nFigure saved → {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Monitor ergodic convergence of the consumption rule"
    )
    p.add_argument("--checkpoint_dir", required=True,
                   help="Root checkpoint directory, e.g. checkpoints/Bewley_Hetero_…")
    p.add_argument("--step", type=int, default=None,
                   help="Which saved step to load (default: model_final.pt)")
    p.add_argument("--config", required=True,
                   help="Path to YAML config used for this run")
    p.add_argument("--n_sim_steps", type=int, default=300,
                   help="Number of simulation steps to run (default: 300)")
    p.add_argument("--snapshot_every", type=int, default=5,
                   help="Record a cross-section snapshot every N steps (default: 5)")
    p.add_argument("--n_focal", type=int, default=10,
                   help="Number of focal agents to track individually (default: 10)")
    p.add_argument("--output", default="convergence.png",
                   help="Output figure path (default: convergence.png)")
    p.add_argument("--n_bins", type=int, default=30,
                   help="Number of bins for c(m) curves (default: 30)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Config ────────────────────────────────────────────────────────
    print(f"Loading config: {args.config}")
    config_dict = load_configs([args.config])
    config_dict  = compute_derived_params(config_dict)
    config       = dict_to_namespace(config_dict)

    device_str = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    device = torch.device(device_str)
    print(f"Device: {device}")

    # ── 2. Load checkpoint ───────────────────────────────────────────────
    print(f"\nLoading checkpoint from: {args.checkpoint_dir}")
    policy_net, normalizer, ckpt_n_agents = load_checkpoint(
        args.checkpoint_dir, args.step, config, device_str
    )

    # Patch config so the environment and initial state use the checkpoint's n_agents
    if ckpt_n_agents != config.training.agents:
        print(f"  [info] Overriding config agents {config.training.agents} → {ckpt_n_agents}")
        config.training.agents = ckpt_n_agents

    # ── 3. Fresh initial state (not from checkpoint) ─────────────────────
    print("\nInitialising fresh agent states from config…")
    main_state = initialize_env_state(config, device)
    print(f"  Batch size : {config.training.batch_size}")
    print(f"  Agents     : {config.training.agents}")
    print(f"  mean(m₀)   : {main_state.moneydisposable.mean():.3f}")
    print(f"  mean(a₀)   : {main_state.savings.mean():.3f}")
    print(f"  mean(v₀)   : {main_state.ability.mean():.3f}")

    # ── 4. Build environment (normalizer is frozen, not trained) ─────────
    env = EconomyEnv(config, normalizer=normalizer, device=device)
    fix_ability = getattr(config.bewley_model, "fix", False)
    print(f"  Market type: {'COMPLETE' if fix_ability else 'INCOMPLETE'}")

    # ── 5. Simulate ───────────────────────────────────────────────────────
    snapshots = run_simulation(
        env=env,
        policy_net=policy_net,
        main_state=main_state,
        n_steps=args.n_sim_steps,
        snapshot_every=args.snapshot_every,
        fix_ability=fix_ability,
        config=config,
    )

    # ── 6. Plot ───────────────────────────────────────────────────────────
    label = os.path.basename(args.checkpoint_dir)
    step_label = f"step={args.step}" if args.step else "final"
    plot_convergence(
        snapshots=snapshots,
        save_path=args.output,
        checkpoint_label=f"{label}  [{step_label}]",
        n_bins=args.n_bins,
    )


if __name__ == "__main__":
    main()
