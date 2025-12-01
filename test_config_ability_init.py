#!/usr/bin/env python3
"""
Test ability initialization FROM CONFIG FILE.

This script tests the initialization specified in your config file,
unlike test_ability_init.py which compares hardcoded methods.

Usage:
    python test_config_ability_init.py                    # Uses config/iq_init.yaml
    python test_config_ability_init.py config/my.yaml    # Uses custom config
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ability_init import initialize_ability, get_ability_init_stats
from src.utils.configloader import load_configs, dict_to_namespace


def plot_distribution(ability, config_path, stats):
    """Plot ability distribution from config."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ability_flat = ability.flatten()

    # 1. Histogram
    ax = axes[0]
    ax.hist(ability_flat, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2,
               label=f"Mean={stats['mean']:.2f}")
    ax.axvline(stats['p50'], color='blue', linestyle='--', linewidth=2,
               label=f"Median={stats['p50']:.2f}")
    ax.set_xlabel('Ability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Ability Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Box plot
    ax = axes[1]
    bp = ax.boxplot(ability_flat, vert=True, widths=0.5, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['medians'][0].set_color('red')
    bp['medians'][0].set_linewidth(2)

    # Add percentile markers
    for percentile, color, label in [(5, 'green', 'P5'),
                                      (25, 'orange', 'P25'),
                                      (75, 'orange', 'P75'),
                                      (95, 'green', 'P95')]:
        val = stats[f'p{percentile}']
        ax.axhline(val, color=color, linestyle=':', linewidth=1.5, alpha=0.7)
        ax.text(1.15, val, label, fontsize=9, color=color)

    ax.set_ylabel('Ability', fontsize=12)
    ax.set_title('Distribution Summary', fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Stats table
    ax = axes[2]
    ax.axis('off')

    stats_text = f"""
    Configuration: {Path(config_path).name}

    Central Tendency:
      Mean:     {stats['mean']:.4f}
      Median:   {stats['p50']:.4f}

    Dispersion:
      Std:      {stats['std']:.4f}
      CV:       {stats['cv']:.4f}

    Range:
      Min:      {stats['min']:.4f}
      Max:      {stats['max']:.4f}

    Percentiles:
      P5:       {stats['p5']:.4f}
      P25:      {stats['p25']:.4f}
      P75:      {stats['p75']:.4f}
      P95:      {stats['p95']:.4f}
      P99:      {stats['p99']:.4f}

    IQR: {stats['p75'] - stats['p25']:.4f}
    P95-P5: {stats['p95'] - stats['p5']:.4f}
    """

    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    return fig


def main():
    # Get config path from command line or use default
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config/iq_init.yaml"

    print("=" * 70)
    print(f"TESTING ABILITY INITIALIZATION FROM: {config_path}")
    print("=" * 70)

    # Load config
    try:
        config_dict = load_configs([config_path])
        config = dict_to_namespace(config_dict)
    except Exception as e:
        print(f"ERROR: Failed to load config from {config_path}")
        print(f"  {e}")
        return

    # Display config settings
    print("\nConfiguration Settings:")
    print("-" * 70)

    if hasattr(config, 'initial_state'):
        init_state = config.initial_state

        # Check if ability init settings exist
        if hasattr(init_state, 'ability_init_method'):
            print(f"  Method:    {init_state.ability_init_method}")

            if init_state.ability_init_method == 'iq_like':
                mean = getattr(init_state, 'ability_mean', 1.0)
                cv = getattr(init_state, 'ability_cv', 0.3)
                min_ab = getattr(init_state, 'ability_min', 0.3)
                max_ab = getattr(init_state, 'ability_max', 3.0)

                print(f"  Mean:      {mean}")
                print(f"  CV:        {cv}")
                print(f"  Min:       {min_ab}")
                print(f"  Max:       {max_ab}")

                # Calculate expected stats
                print("\n  Expected Distribution:")
                print(f"    Std ≈ Mean × CV = {mean * cv:.4f}")
                print(f"    Most agents in: [{mean * 0.5:.2f}, {mean * 1.7:.2f}]")

        else:
            print("  WARNING: No ability_init_method in config!")
            print("  Using default: iq_like with mean=1.0, cv=0.3")

    # Parameters
    batch_size = 64
    n_agents = 800

    print(f"\nInitializing ability for:")
    print(f"  Batch size: {batch_size}")
    print(f"  N agents:   {n_agents}")
    print(f"  Total:      {batch_size * n_agents} samples")

    # Initialize ability using config
    print("\n" + "=" * 70)
    print("INITIALIZING ABILITY")
    print("=" * 70)

    init_method = getattr(config.initial_state, 'ability_init_method', 'iq_like')

    if init_method == 'iq_like':
        ability = initialize_ability(
            batch_size, n_agents,
            method='iq_like',
            mean=getattr(config.initial_state, 'ability_mean', 1.0),
            cv=getattr(config.initial_state, 'ability_cv', 0.3),
            min_ability=getattr(config.initial_state, 'ability_min', 0.3),
            max_ability=getattr(config.initial_state, 'ability_max', 3.0)
        )
    elif init_method == 'stationary':
        ability = initialize_ability(
            batch_size, n_agents,
            method='stationary',
            config=config,
            clip_sigma=2.0
        )
    elif init_method == 'narrow_uniform':
        ability = initialize_ability(
            batch_size, n_agents,
            method='narrow_uniform',
            mean=getattr(config.initial_state, 'ability_mean', 1.0),
            half_width=getattr(config.initial_state, 'ability_half_width', 0.2)
        )
    else:
        print(f"ERROR: Unknown init method: {init_method}")
        return

    # Compute stats
    stats = get_ability_init_stats(ability)

    print("\n✓ Initialization complete!")
    print("\nActual Distribution:")
    print("-" * 70)
    print(f"  Mean:      {stats['mean']:.4f}")
    print(f"  Std:       {stats['std']:.4f}")
    print(f"  CV:        {stats['cv']:.4f}")
    print(f"  Min:       {stats['min']:.4f}")
    print(f"  Max:       {stats['max']:.4f}")
    print(f"  P5-P95:    [{stats['p5']:.4f}, {stats['p95']:.4f}]")
    print(f"  P25-P75:   [{stats['p25']:.4f}, {stats['p75']:.4f}]")

    # Check if stats match config
    if init_method == 'iq_like':
        expected_mean = getattr(config.initial_state, 'ability_mean', 1.0)
        expected_cv = getattr(config.initial_state, 'ability_cv', 0.3)

        mean_error = abs(stats['mean'] - expected_mean) / expected_mean
        cv_error = abs(stats['cv'] - expected_cv) / expected_cv

        print("\n  Match with config:")
        print(f"    Mean error: {mean_error*100:.2f}%  {'✓' if mean_error < 0.05 else '⚠'}")
        print(f"    CV error:   {cv_error*100:.2f}%  {'✓' if cv_error < 0.1 else '⚠'}")

    # Generate plot
    print("\n" + "=" * 70)
    print("GENERATING PLOT")
    print("=" * 70)

    fig = plot_distribution(ability, config_path, stats)

    # Save with config-specific name
    config_name = Path(config_path).stem
    output_path = f"test_ability_init_{config_name}.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    plt.close()

    # Warnings
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    warnings = []

    if stats['cv'] > 0.8:
        warnings.append("⚠ Very high CV (>0.8): May cause instability")

    if stats['max'] / stats['mean'] > 10:
        warnings.append("⚠ Max/Mean ratio > 10: Extreme outliers present")

    if stats['min'] < 0.1:
        warnings.append("⚠ Very low minimum (<0.1): Near-zero ability may cause issues")

    # Check bound violations
    if init_method == 'iq_like':
        expected_min = getattr(config.initial_state, 'ability_min', 0.3)
        expected_max = getattr(config.initial_state, 'ability_max', 3.0)

        at_min = np.sum(ability == expected_min)
        at_max = np.sum(ability == expected_max)
        total = ability.size

        if at_min > total * 0.01:
            warnings.append(f"⚠ {at_min/total*100:.1f}% of agents hit minimum bound")

        if at_max > total * 0.01:
            warnings.append(f"⚠ {at_max/total*100:.1f}% of agents hit maximum bound")

    if warnings:
        for warning in warnings:
            print(warning)
    else:
        print("✓ All checks passed!")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_path}")
    print(f"Config used: {config_path}")


if __name__ == "__main__":
    main()
