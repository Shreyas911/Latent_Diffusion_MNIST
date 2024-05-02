import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image, make_grid

import numpy as np

import os
import sys
# Append to sys.path the absolute path to src/Latent_Diffusion_MNIST
path_list = os.path.abspath('').split('/')
path_src_Latent_Diffusion_MNIST = ''
for link in path_list[:-1]:
    path_src_Latent_Diffusion_MNIST = path_src_Latent_Diffusion_MNIST+link+'/'
sys.path.append(path_src_Latent_Diffusion_MNIST+'/src')

# Now import module Latent_Diffusion_MNIST
from Latent_Diffusion_MNIST import *

if __name__ == "__main__":

    # Fix the seed
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # Enable CuDNN benchmarking for optimized performance
    torch.backends.cudnn.benchmark = True

    # Define data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.5,), (1.0,))  # Normalize the pixel values to range [-1, 1]
    ])

    # Load the training set
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    # num_samples_to_load = 128
    # subset_indices = torch.randperm(len(train_dataset))[:num_samples_to_load]
    # subset_dataset = Subset(train_dataset, subset_indices)

    dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=20)

    unet = UNetVanilla(n_channel = 1)
    ddpm = DDPMNoTimeEmbedding(nn_eps = unet,
                               betas  = (1.e-4,0.02),
                               n_T    = 1000,
                               beta_schedule = linear_schedule,
                               criterion = nn.MSELoss())
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

    trainer = Trainer(model        = ddpm,
                      train_loader = dataloader,
                      optimizer    = optim, 
                      n_epochs     = 100,
                      model_name   = './ddpm_mnist_new.pth')

    trainer.train()