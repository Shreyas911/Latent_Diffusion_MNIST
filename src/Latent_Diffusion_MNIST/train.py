from beartype import beartype
from beartype.typing import Callable, Type

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image, make_grid

from tqdm import tqdm

from Latent_Diffusion_MNIST.networks import *
from Latent_Diffusion_MNIST.models import *

__all__ = ["Trainer"]

class Trainer:
    
    @beartype
    def __init__(self,
                 *,
                 model: nn.Module,
                 train_loader: DataLoader,
                 optimizer: torch.optim.Optimizer, 
                 n_epochs: int,
                 model_name: str) -> None:

        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.model_name = model_name

    @beartype
    def train(self) -> None:

        for i in range(self.n_epochs):
            self.model.train()
            pbar = tqdm(self.train_loader)
            loss_ema = None
            for x, _ in pbar:
                self.optimizer.zero_grad()
                loss = self.model(x)
                loss.backward()
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
                pbar.set_description(f"loss: {loss_ema:.4f}")
                self.optimizer.step()

            self.model.eval()

            # save model
            torch.save(self.model.state_dict(), f"./{self.model_name}_mnist.pth")

            with torch.no_grad():
                xh = self.model.sample(n_samples = 16, 
                                       size      = (1, 28, 28))
                grid = make_grid(xh, nrow=4)
                save_image(grid, f"./contents/{self.model_name}_sample_{i}.png")
                # grid_np = TF.to_pil_image(grid)
                # plt.imshow(grid_np)
                # plt.axis('off')
                # plt.show()