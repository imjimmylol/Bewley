"""
Diagnostic script: money_disposable distribution by v_bar quantile.

Uses dummy (fixed) labor to isolate the effect of heterogeneous v_bar
on money_disposable. Groups agents by v_bar quantile (q10, q25, q50, q75, q90)
and plots each group's money_disposable distribution separately.

Usage:
    python test/test_money_disposable_distribution.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.utils.configloader import load_configs, compute_derived_params, dict_to_namespace
from src.environment import EconomyEnv
from src.train import initialize_env_state
from src.shocks import transition_ability

# ── Config ──────────────────────────────────────────────────────────────────
CONFIG_PATH = "config/0303/iq_init_incomplete_notax_low_uncer.yaml"
DUMMY_LABOR = 0.5          # Fixed labor for all agents
N_TRANSITION_STEPS = 10000    # AR(1) steps to let ability diverge by v_bar
SAVE_PATH = "test/fig_money_disposable_by_vbar.png"

# ── Setup ───────────────────────────────────────────────────────────────────
cfg_dict = load_configs([CONFIG_PATH])
cfg_dict = compute_derived_params(cfg_dict)
config = dict_to_namespace(cfg_dict)

device = torch.device("cpu")
env = EconomyEnv(config, normalizer=None, device=device)
main_state = initialize_env_state(config, device)

B = config.training.batch_size
A = config.training.agents

print(f"Batch size: {B}, Agents: {A}")
print(f"v_bar shape: {main_state.v_bar.shape}")
print(f"v_bar range: [{main_state.v_bar.min():.2f}, {main_state.v_bar.max():.2f}]")
print(f"v_bar mean:  {main_state.v_bar.mean():.2f}")

# ── Run AR(1) transitions to let ability diverge ────────────────────────────
# Start ability at v_bar (each agent's own long-run mean)
ability = main_state.v_bar.clone()
is_superstar = torch.zeros(B, A, dtype=torch.bool, device=device)
v_bar = main_state.v_bar

for t in range(N_TRANSITION_STEPS):
    ability, is_superstar = transition_ability(
        ability_t=ability,
        is_superstar_t=is_superstar,
        rho_v=config.shock.rho_v,
        sigma_v=config.shock.sigma_v,
        v_bar=v_bar,
        v_min=config.shock.v_min,
        v_max=config.shock.v_max,
        p=config.shock.p,
        q=config.shock.q,
    )

print(f"\nAfter {N_TRANSITION_STEPS} AR(1) steps:")
print(f"ability range: [{ability.min():.2f}, {ability.max():.2f}]")
print(f"ability mean:  {ability.mean():.2f}")

# ── Compute money_disposable with dummy labor ───────────────────────────────
savings = main_state.savings
labor = torch.full((B, A), DUMMY_LABOR, device=device)

wage, ret = env._compute_market_equilibrium(
    savings=savings,
    labor=labor,
    ability=ability,
    A=config.bewley_model.A,
    alpha=config.bewley_model.alpha,
)

income_outcomes = env._compute_income_tax(
    wage=wage,
    labor=labor,
    ability=ability,
    savings=savings,
    ret_lagged=main_state.ret,
)

md = income_outcomes["money_disposable"]   # (B, A)
ibt = income_outcomes["income_before_tax"]

print(f"\nmoney_disposable range: [{md.min():.2f}, {md.max():.2f}]")
print(f"money_disposable mean:  {md.mean():.2f}")

# ── Group agents by v_bar quantile ──────────────────────────────────────────
# Flatten across batches for plotting: treat each (batch, agent) as a sample
v_bar_flat = v_bar.reshape(-1).numpy()
md_flat = md.detach().reshape(-1).numpy()
ability_flat = ability.detach().reshape(-1).numpy()

quantile_edges = [0, 10, 25, 50, 75, 90, 100]
quantile_labels = ["q0-q10", "q10-q25", "q25-q50", "q50-q75", "q75-q90", "q90-q100"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
percentiles = np.percentile(v_bar_flat, quantile_edges)

groups = {}
for i in range(len(quantile_labels)):
    lo, hi = percentiles[i], percentiles[i + 1]
    if i == len(quantile_labels) - 1:
        mask = (v_bar_flat >= lo) & (v_bar_flat <= hi)
    else:
        mask = (v_bar_flat >= lo) & (v_bar_flat < hi)
    groups[quantile_labels[i]] = {
        "md": md_flat[mask],
        "ability": ability_flat[mask],
        "v_bar_range": (lo, hi),
        "count": mask.sum(),
    }

# ── Print summary table ────────────────────────────────────────────────────
print(f"\n{'Group':<12} {'v_bar range':<20} {'N':>6} {'md mean':>10} {'md std':>10} {'ability mean':>12}")
print("-" * 72)
for label, g in groups.items():
    print(f"{label:<12} [{g['v_bar_range'][0]:.2f}, {g['v_bar_range'][1]:.2f}]{'':<6} "
          f"{g['count']:>6} {g['md'].mean():>10.2f} {g['md'].std():>10.2f} {g['ability'].mean():>12.2f}")

# ── Plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: money_disposable histograms by v_bar quantile
ax = axes[0]
for i, (label, g) in enumerate(groups.items()):
    lo, hi = g["v_bar_range"]
    ax.hist(
        g["md"], bins=40, alpha=0.45, color=colors[i],
        label=f"{label} (v̄∈[{lo:.1f},{hi:.1f}])", density=True,
    )
ax.set_xlabel("money_disposable")
ax.set_ylabel("Density")
ax.set_title(f"money_disposable by v̄ quantile\n(labor={DUMMY_LABOR}, {N_TRANSITION_STEPS} AR(1) steps)")
ax.legend(fontsize=7)
ax.axvline(md_flat.mean(), color="k", ls="--", lw=0.8, label="overall mean")

# Panel 2: ability histograms by v_bar quantile (sanity check)
ax = axes[1]
for i, (label, g) in enumerate(groups.items()):
    lo, hi = g["v_bar_range"]
    ax.hist(
        g["ability"], bins=40, alpha=0.45, color=colors[i],
        label=f"{label}", density=True,
    )
ax.set_xlabel("ability (after transitions)")
ax.set_ylabel("Density")
ax.set_title(f"Ability by v̄ quantile\n(sanity check: higher v̄ → higher ability)")
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=150)
print(f"\nPlot saved to {SAVE_PATH}")
plt.show()
