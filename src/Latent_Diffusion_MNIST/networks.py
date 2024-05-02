from beartype import beartype

import torch
import torch.nn as nn

__all__ = ["UNetVanilla"]

class UNetVanilla(nn.Module):
    
    @beartype
    def __init__(self, *, n_channel: int = 1) -> None:
        super().__init__()

        self.conv_block = lambda ic, oc: nn.Sequential(nn.Conv2d(ic, oc, 7, padding=3),
                                                       nn.BatchNorm2d(oc),
                                                       nn.LeakyReLU(),
                                                      )
        self.n_channel = n_channel
        self.conv = nn.Sequential(self.conv_block(self.n_channel, 64),
                                  self.conv_block(64, 128),
                                  self.conv_block(128, 256),
                                  self.conv_block(256, 512),
                                  self.conv_block(512, 256),
                                  self.conv_block(256, 128),
                                  self.conv_block(128, 64),
                                  nn.Conv2d(64, n_channel, 3, padding=1),)

    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)