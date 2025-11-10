# src/utils/configloader.py
import yaml
import math
from types import SimpleNamespace
from collections.abc import Mapping
from functools import reduce

def deep_merge(d1, d2):
    """
    Recursively merges d2 into a copy of d1.
    """
    d1 = d1.copy()
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], Mapping) and isinstance(v, Mapping):
            d1[k] = deep_merge(d1[k], v)
        else:
            d1[k] = v
    return d1

def dict_to_namespace(d):
    """Recursively convert a dict into a SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d

def compute_derived_params(cfg_dict):
    """
    Compute derived parameters and add them to the config dict.
    This function operates on the dictionary before namespace conversion.
    """
    if 'shock' in cfg_dict and 'rho_v' in cfg_dict['shock'] and 'sigma_v' in cfg_dict['shock']:
        try:
            rho_v = cfg_dict['shock']['rho_v']
            sigma_v = cfg_dict['shock']['sigma_v']
            # Avoid math domain error for rho_v >= 1.0
            if abs(rho_v) < 1.0:
                denom = math.sqrt(1 - rho_v ** 2)
                cfg_dict['shock']['v_min'] = math.exp(-2 * sigma_v / denom)
                cfg_dict['shock']['v_max'] = math.exp(2 * sigma_v / denom)
            else:
                cfg_dict['shock']['v_min'] = 0
                cfg_dict['shock']['v_max'] = float('inf')

        except (TypeError, KeyError) as e:
            print(f"Warning: Could not compute shock bounds. Missing key or wrong type in config: {e}")

    # The logic from SML-2 to convert tax_params to a dict is not needed here,
    # as we will keep it as a namespace for attribute access.
    # If a dictionary is truly needed downstream, it can be converted there.

    return cfg_dict

def load_configs(paths: list[str]) -> dict:
    """Load multiple YAML configs, merge them, and return as a dict."""
    configs = []
    for path in paths:
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
                if config:
                    configs.append(config)
        except FileNotFoundError:
            print(f"Warning: Config file not found at {path}, skipping.")
        except yaml.YAMLError as e:
            print(f"Warning: Error parsing YAML file {path}: {e}")

    if not configs:
        return {}

    # Merge all loaded configs. The `reduce` function applies `deep_merge` cumulatively.
    merged_config = reduce(deep_merge, configs)

    # Compute derived parameters on the final merged dict
    merged_config = compute_derived_params(merged_config)

    return merged_config
