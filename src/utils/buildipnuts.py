import torch 

def build_inputs(moneydisposable, ability, tax_params, device, eps=1e-8):
    """
    回傳:
      features : (B, A, 2A + 2)   # 給模型輸入
      condi    : (B, A, Z)        # 稅制條件
    """
    # ---- 1. 保證都是 tensor ----
    moneydisposable = torch.as_tensor(moneydisposable, dtype=torch.float32, device=device)
    ability         = torch.as_tensor(ability,         dtype=torch.float32, device=device)
    tax_params      = torch.as_tensor(tax_params,      dtype=torch.float32, device=device)

    B, A = moneydisposable.shape
    
    condi = tax_params.unsqueeze(1).expand(-1, A, -1)   # tax_params: (B, Z)

    # (B, 2A) -> (B, A, 2A)
    sum_info = torch.cat([moneydisposable, ability], dim=1)     # (B, 2A)
    sum_info_rep = sum_info.unsqueeze(1).expand(-1, A, -1)      # (B, A, 2A)

    # (B, A, 1) × 2
    money_self   = moneydisposable.unsqueeze(-1)  # (B, A, 1)
    ability_self = ability.unsqueeze(-1)          # (B, A, 1)

    features = torch.cat([sum_info_rep, money_self, ability_self], dim=2)  # (B, A, 2A+2)

    return features, condi