from beartype import beartype

import torch
import torch.nn as nn

__all__ = ["UNetVanilla", "ContextUNet"]

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

class ResidualConvBlock(nn.Module):

    @beartype
    def __init__(self, *, 
                 input_channels: int,
                 output_channels: int,
                 is_res: bool = False) -> None:
        super().__init__()

        self.same_channels = input_channels == output_channels
        self.is_res        = is_res
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 1, 1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.GELU(),)
        self.conv2 = nn.Sequential(nn.Conv2d(output_channels, output_channels, 3, 1, 1),
                                   nn.BatchNorm2d(output_channels),
                                   nn.GELU(),)

    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        if self.is_res:
            if self.same_channels:
                return x + x2
            else:
                return x1 + x2
        else:
            return x2

class UNetDown(nn.Module):

    @beartype
    def __init__(self, *, 
                 input_channels: int,
                 output_channels: int,
                 is_res: bool = False) -> None:
        super().__init__()
        self.model = nn.Sequential(
                    ResidualConvBlock(input_channels = input_channels, 
                                      output_channels = output_channels, 
                                      is_res = is_res),
                    nn.MaxPool2d(2))

    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

class UNetUp(nn.Module):

    @beartype
    def __init__(self, *, 
                 input_channels: int,
                 output_channels: int,
                 is_res: bool = False) -> None:
        super().__init__()
        self.model = nn.Sequential(
                    nn.ConvTranspose2d(input_channels, output_channels, 2, 2),
                    ResidualConvBlock(input_channels = output_channels, 
                                      output_channels = output_channels, 
                                      is_res = is_res),
                    ResidualConvBlock(input_channels = output_channels, 
                                      output_channels = output_channels, 
                                      is_res = is_res),)

    @beartype
    def forward(self, 
                x: torch.Tensor, 
                skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, skip), 1)
        return self.model(x)

class EmbedFC(nn.Module):

    @beartype
    def __init__(self, *,
                 input_dim: int,
                 embed_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(nn.Linear(input_dim, embed_dim),
                                   nn.GELU(),
                                   nn.Linear(embed_dim, embed_dim),)
    
    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_dim)
        return self.model(x)

class ContextUNet(nn.Module):

    @beartype
    def __init__(self, *, 
                 in_channels: int = 1, 
                 n_feat: int = 256, 
                 n_classes: int = 10,
                 is_res: bool = False):
        super().__init__()
    
        self.in_channels = in_channels
        self.n_feat      = n_feat
        self.n_classes   = n_classes

        self.conv0 = ResidualConvBlock(input_channels = in_channels,   
                                       output_channels = n_feat,
                                       is_res = is_res)
        self.down1 = UNetDown(input_channels = n_feat, 
                              output_channels = n_feat, 
                              is_res = is_res)
        self.down2 = UNetDown(input_channels = n_feat, 
                              output_channels = 2*n_feat, 
                              is_res = is_res)
        self.pool0 = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(input_dim = 1, 
                                  embed_dim = 2*n_feat)
        self.timeembed2 = EmbedFC(input_dim = 1, 
                                  embed_dim = 1*n_feat)
        self.contextembed1 = EmbedFC(input_dim = n_classes, 
                                     embed_dim = 2*n_feat)
        self.contextembed2 = EmbedFC(input_dim = n_classes, 
                                     embed_dim = 1*n_feat)

        self.up0 = nn.Sequential(
            # If time and context embeddings are concatenated instead of multiplied and added, can have 
            # nn.ConvTranspose2d(6*n_feat, 2*n_feat, 7, 7)
            nn.ConvTranspose2d(2*n_feat, 2*n_feat, 7, 7),
            nn.GroupNorm(8, 2*n_feat),
            nn.ReLU(),)

        self.up1 = UNetUp(input_channels = 4*n_feat, 
                          output_channels = n_feat, 
                          is_res = is_res)
        self.up2 = UNetUp(input_channels = 2*n_feat, 
                          output_channels = n_feat, 
                          is_res = is_res)

        self.out = nn.Sequential(nn.Conv2d(2*n_feat, n_feat, 3, 1, 1),
                                 nn.GroupNorm(8, n_feat),
                                 nn.ReLU(),
                                 nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),)

    @beartype
    def forward(self,
                x: torch.Tensor, 
                c: torch.Tensor, 
                t: torch.Tensor, 
                context_mask: torch.Tensor) -> torch.Tensor:

        x = self.conv0(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        pool0 = self.pool0(down2)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        c = c * context_mask
        
        timeembed1    = self.timeembed1(t).view(-1, 2*self.n_feat, 1, 1)
        timeembed2    = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        contextembed1 = self.contextembed1(c).view(-1, 2*self.n_feat, 1, 1)
        contextembed2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(pool0)
        up2 = self.up1(contextembed1*up1 + timeembed1, down2)  # add and multiply embeddings
        up3 = self.up2(contextembed2*up2 + timeembed2, down1)
        out = self.out(torch.cat((up3, x), 1))

        return out


    