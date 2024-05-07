from beartype import beartype
from beartype.typing import Callable, Dict, Tuple, Type

import torch
import torch.nn as nn

from Latent_Diffusion_MNIST.networks import *

__all__ = ["DDPMNoTimeEmbedding", 
           "DDIMNoTimeEmbedding", 
           "DDPMTimeContextEmbed"]

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

        super().__init__(nn_eps        = nn_eps,
                         betas         = betas,
                         n_T           = n_T,
                         beta_schedule = beta_schedule,
                         criterion     = criterion)

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

        xs = torch.randn(n_samples, *size)

        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_samples, *size) if i > 1 else 0
            eps = self.nn_eps(xs)
            x0_t = (xs - eps * (1 - torch.sqrt(self.alphabar_t[i]))) / torch.sqrt(self.alphabar_t[i])
            c1 = self.eta * ((1 - self.alphabar_t[i] / self.alphabar_t[i - 1]) * (1 - self.alphabar_t[i - 1]) / (
                    1 - self.alphabar_t[i])).sqrt()
            c2 = ((1 - self.alphabar_t[i - 1]) - c1 ** 2).sqrt()
            xs = self.alphabar_t[i - 1].sqrt() * x0_t + c1 * z + c2 * eps

        return xs

class DDPMTimeContextEmbed(DDPMNoTimeEmbedding):

    @beartype
    def __init__(self,
                 *,
                 nn_eps: nn.Module,
                 betas: Tuple[float, float], 
                 n_T: int, 
                 beta_schedule: Callable[[float, float, int], Dict],
                 criterion: nn.Module = nn.MSELoss(),
                 drop_prob: float = 0.1) -> None:

        super().__init__(nn_eps        = nn_eps,
                         betas         = betas,
                         n_T           = n_T,
                         beta_schedule = beta_schedule,
                         criterion     = criterion)

        self.drop_prob = 0.1
        for k, v in beta_schedule(beta1 = betas[0], 
                                  beta2 = betas[1], 
                                  T     = n_T).items():
            self.register_buffer(k, v)

    @beartype
    def forward(self, x: torch.Tensor, c:torch.Tensor) -> torch.Tensor:
        _ts = torch.randint(0, self.n_T, (x.shape[0],))
        eps = torch.randn_like(x)
        x_t = self.sqrtab[_ts, None, None, None] * x +  self.sqrtmab[_ts, None, None, None] * eps

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob)
        
        # return MSE between added noise, and our predicted noise
        return self.criterion(eps, self.nn_eps(x_t, c, _ts / self.n_T, context_mask))

    @torch.no_grad()
    @beartype
    def sample(self,
               *,
               n_samples: int, 
               size: Tuple[int, int, int],
               guide_w: float = 0.0) -> torch.Tensor:

        xs = torch.randn(n_samples, *size)
        cs = torch.arange(0,10)
        cs = cs.repeat(int(n_samples/c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(cs)

        # double the batch
        cs = cs.repeat(2)
        context_masks = context_masks.repeat(2)
        context_masks[n_samples:] = 1. # makes second half of batch context free

        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_samples, *size) if i > 1 else 0
            ts = torch.tensor([i / self.n_T])
            ts = ts.repeat(n_samples)
            eps = self.nn_eps(xs, cs, ts, context_masks)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            xs = self.oneover_sqrta[i] * (xs - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z

        return xs

        