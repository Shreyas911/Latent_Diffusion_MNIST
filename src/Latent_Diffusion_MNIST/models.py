from beartype import beartype
from beartype.typing import Callable, Dict, Tuple, Type

import torch
import torch.nn as nn

from Latent_Diffusion_MNIST.networks import *

__all__ = ["DDPMNoTimeEmbedding"]

class DDPMNoTimeEmbedding(nn.Module):

    @beartype
    def __init__(self,
                 *,
                 nn_eps: nn.Module,
                 betas: Tuple[float, float], 
                 n_T: int, 
                 beta_schedule: Callable[[float, float, int], Dict],
                 criterion: nn.Module = nn.MSELoss()) -> None:

        super().__init__()
        self.nn_eps = nn_eps
        self.betas = betas
        self.n_T = n_T
        self.criterion = criterion

        for k, v in beta_schedule(beta1 = betas[0], 
                                  beta2 = betas[1], 
                                  T     = n_T).items():
            self.register_buffer(k, v)

    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ts = torch.randint(0, self.n_T, (x.shape[0],))
        eps = torch.randn_like(x)
        x_t = self.sqrtab[_ts, None, None, None] * x +  self.sqrtmab[_ts, None, None, None] * eps

        return self.criterion(eps, self.nn_eps(x_t))

    @torch.no_grad()
    @beartype
    def sample(self,
               *,
               n_samples: int, 
               size: Tuple[int, int, int]) -> torch.Tensor:

        xs = torch.randn(n_samples, *size)

        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_samples, *size) if i > 1 else 0
            # eps = self.nn_eps(xs, i / self.n_T)
            eps = self.nn_eps(xs)
            xs = self.oneover_sqrta[i] * (xs - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z

        return xs

class DDIMNoTimeEmbedding(DDPMNoTimeEmbedding):

    @beartype
    def __init__(self,
                 *,
                 nn_eps: nn.Module,
                 betas: Tuple[float, float], 
                 n_T: int, 
                 beta_schedule: Callable[[float, float, int], Dict],
                 criterion: nn.Module = nn.MSELoss()) -> None:

        super().__init__(nn_eps = nn_eps,
                         betas  = betas,
                         n_T    = n_T,
                         criterion = criterion)

        for k, v in beta_schedule(beta1 = betas[0], 
                                  beta2 = betas[1], 
                                  T     = n_T).items():
            self.register_buffer(k, v)

    @torch.no_grad()
    @beartype
    def sample(self,
               *,
               n_samples: int, 
               size: Tuple[int, int, int]) -> torch.Tensor:

        pass