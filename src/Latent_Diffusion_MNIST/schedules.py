from beartype import beartype
from beartype.typing import Dict

import torch
import torch.nn as nn

__all__ = ["linear_schedule"]

def linear_schedule(*,
                    beta1: float, 
                    beta2: float, 
                    T: int) -> Dict:

    assert 0.0 < beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
    assert 0 < T, "T should be greater than 0"

    beta_t = beta1 + torch.arange(0, T + 1, dtype=torch.float32) / T * (beta2 - beta1)
    sqrt_beta_t = torch.sqrt(beta_t)
    
    alpha_t = 1 - beta_t
    oneover_sqrta = 1/torch.sqrt(alpha_t)

    alphabar_t = torch.cumsum(torch.log(alpha_t), dim = 0).exp()
    sqrtab = torch.sqrt(alphabar_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab = (1 - alpha_t) / sqrtmab

    return {'beta_t': beta_t,
            "sqrt_beta_t": sqrt_beta_t,
            "alpha_t": alpha_t, 
            "oneover_sqrta": oneover_sqrta,
            "alphabar_t": alphabar_t,
            "sqrtab": sqrtab,
            "sqrtmab": sqrtmab,
            "mab_over_sqrtmab": mab_over_sqrtmab}