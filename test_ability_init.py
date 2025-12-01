#!/usr/bin/env python3
"""
Test script for ability initialization strategies.

Compares different initialization methods:
1. Stationary AR(1) (old method) - can explode over time
2. IQ-like distribution (RECOMMENDED) - bounded realistic distribution
3. Narrow uniform (testing) - minimal heterogeneity

Run:
    python test_ability_init.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ability_init import (
    initialize_ability,
    get_ability_init_stats
)
from src.utils.configloader import load_configs, dict_to_namespace


def plot_comparison(abilities_dict, title="Ability Distribution Comparison"):
    """
    Plot comparison of different initialization methods.

    Args:
        abilities_dict: Dict mapping method name to ability array
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)

    for idx, (method, ability) in enumerate(abilities_dict.items()):
        row = idx // 3
        col = idx % 3

        if idx >= 6:
            break

        ax = axes[row, col]

        # Flatten for histogram
        ability_flat = ability.flatten()

        # Histogram
        ax.hist(ability_flat, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(ability_flat), color='red', linestyle='--',
                   linewidth=2, label=f'Mean={np.mean(ability_flat):.2f}')
        ax.axvline(np.median(ability_flat), color='blue', linestyle='--',
                   linewidth=2, label=f'Median={np.median(ability_flat):.2f}')

        # Stats
        stats = get_ability_init_stats(ability)
        stats_text = (
            f"Mean: {stats['mean']:.3f}\n"
            f"Std: {stats['std']:.3f}\n"
            f"CV: {stats['cv']:.3f}\n"
            f"Min: {stats['min']:.3f}\n"
            f"Max: {stats['max']:.3f}\n"
            f"P5-P95: [{stats['p5']:.2f}, {stats['p95']:.2f}]"
        )

        ax.text(0.98, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

        ax.set_xlabel('Ability')
        ax.set_ylabel('Count')
        ax.set_title(method)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(abilities_dict), 6):
        row = idx // 3
        col = idx % 3
        axes[row, col].axis('off')

    plt.tight_layout()
    return fig


def main():
    print("=" * 70)
    print("ABILITY INITIALIZATION COMPARISON")
    print("=" * 70)

    # Load config
    config_dict = load_configs(["config/iq_init.yaml"])
    config = dict_to_namespace(config_dict)
    print(f"\nConfig loaded from: config/default.yaml")
    print(f"  rho_v: {config.shock.rho_v}")
    print(f"  sigma_v: {config.shock.sigma_v}")
    print(f"  v_bar: {config.shock.v_bar}")

    # Parameters
    batch_size = 4
    n_agents = 128

    abilities = {}

    # Method 1: Stationary AR(1) - OLD METHOD (can explode)
    print("\n" + "=" * 70)
    print("METHOD 1: Stationary AR(1) (Old Method)")
    print("=" * 70)
    print("WARNING: This can lead to explosive dispersion over time!")

    ability_stationary = initialize_ability(
        batch_size, n_agents,
        method="stationary",
        config=config,
        clip_sigma=2.0  # Clip to ±2σ
    )
    abilities["1. Stationary AR(1)\n(clip_sigma=2.0)"] = ability_stationary
    stats = get_ability_init_stats(ability_stationary)
    print(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}, CV: {stats['cv']:.3f}")
    print(f"Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"P5-P95: [{stats['p5']:.3f}, {stats['p95']:.3f}]")

    # Method 2: IQ-like - RECOMMENDED (conservative)
    print("\n" + "=" * 70)
    print("METHOD 2a: IQ-Like Distribution (Conservative) - RECOMMENDED")
    print("=" * 70)
    print("Like real IQ: most people clustered near mean, moderate tail")

    ability_iq_conservative = initialize_ability(
        batch_size, n_agents,
        method="iq_like",
        mean=1.0,
        cv=0.25,  # Conservative dispersion (like IQ: cv=0.15)
        min_ability=0.4,  # 40% of mean
        max_ability=2.5   # 250% of mean
    )
    abilities["2a. IQ-like (Conservative)\ncv=0.25, range=[0.4, 2.5]"] = ability_iq_conservative
    stats = get_ability_init_stats(ability_iq_conservative)
    print(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}, CV: {stats['cv']:.3f}")
    print(f"Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"P5-P95: [{stats['p5']:.3f}, {stats['p95']:.3f}]")

    # Method 3: IQ-like - Moderate dispersion
    print("\n" + "=" * 70)
    print("METHOD 2b: IQ-Like Distribution (Moderate)")
    print("=" * 70)
    print("Like earnings distribution: moderate inequality")

    ability_iq_moderate = initialize_ability(
        batch_size, n_agents,
        method="iq_like",
        mean=1.0,
        cv=0.4,  # Moderate dispersion (like earnings)
        min_ability=0.3,
        max_ability=3.0
    )
    abilities["2b. IQ-like (Moderate)\ncv=0.4, range=[0.3, 3.0]"] = ability_iq_moderate
    stats = get_ability_init_stats(ability_iq_moderate)
    print(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}, CV: {stats['cv']:.3f}")
    print(f"Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"P5-P95: [{stats['p5']:.3f}, {stats['p95']:.3f}]")

    # Method 4: IQ-like - High inequality
    print("\n" + "=" * 70)
    print("METHOD 2c: IQ-Like Distribution (High Inequality)")
    print("=" * 70)
    print("Like wealth distribution: high inequality, large tail")

    ability_iq_high = initialize_ability(
        batch_size, n_agents,
        method="iq_like",
        mean=1.0,
        cv=0.8,  # High dispersion (like wealth)
        min_ability=0.2,
        max_ability=5.0
    )
    abilities["2c. IQ-like (High Inequality)\ncv=0.8, range=[0.2, 5.0]"] = ability_iq_high
    stats = get_ability_init_stats(ability_iq_high)
    print(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}, CV: {stats['cv']:.3f}")
    print(f"Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"P5-P95: [{stats['p5']:.3f}, {stats['p95']:.3f}]")

    # Method 5: Narrow uniform - for testing
    print("\n" + "=" * 70)
    print("METHOD 3: Narrow Uniform (Testing/Debugging)")
    print("=" * 70)
    print("Minimal heterogeneity to isolate other mechanisms")

    ability_narrow = initialize_ability(
        batch_size, n_agents,
        method="narrow_uniform",
        mean=1.0,
        half_width=0.15  # [0.85, 1.15]
    )
    abilities["3. Narrow Uniform\nmean=1.0, range=[0.85, 1.15]"] = ability_narrow
    stats = get_ability_init_stats(ability_narrow)
    print(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}, CV: {stats['cv']:.3f}")
    print(f"Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"P5-P95: [{stats['p5']:.3f}, {stats['p95']:.3f}]")

    # Plot comparison
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    fig = plot_comparison(abilities)
    output_path = "test_ability_init_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    plt.close()

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. FOR MOST USES: Use 'iq_like' with moderate dispersion
   - initialize_ability(B, A, method='iq_like', mean=1.0, cv=0.3,
                        min_ability=0.3, max_ability=3.0)
   - Realistic, bounded, prevents model collapse
   - CV=0.3 gives moderate heterogeneity

2. FOR CONSERVATIVE ANALYSIS: Use lower CV
   - cv=0.15-0.25 (like real IQ distribution)
   - Easier to see behavior patterns with less noise

3. FOR HIGH INEQUALITY: Use higher CV
   - cv=0.6-0.8 (like wealth distribution)
   - Study redistribution, superstar effects

4. FOR TESTING/DEBUGGING: Use 'narrow_uniform'
   - Minimal heterogeneity isolates other mechanisms
   - Easier to verify model correctness

5. AVOID: Unclipped stationary initialization
   - Current AR(1) parameters (rho=0.98, sigma=0.36) lead to explosion
   - If using, must clip to ±2σ or adjust parameters

KEY INSIGHT: Your current shock parameters are TOO PERSISTENT
   - rho_v=0.98 means shocks persist for ~50 periods (1/(1-0.98))
   - sigma_v=0.36 is large relative to mean
   - Combined: stationary std = 2.55 (huge compared to mean=5)
   - Suggests: Either (a) use bounded init like 'iq_like', or
              (b) reduce rho_v to 0.9 and sigma_v to 0.1
""")


if __name__ == "__main__":
    main()
