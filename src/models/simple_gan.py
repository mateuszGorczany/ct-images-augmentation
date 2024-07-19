import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

# from monai.transforms import (
#     Compose,
#     LoadImage,
#     AddChannel,
#     ScaleIntensity,
#     ToTensor,
#     Resize,
# )
# from monai.data import DataLoader, Dataset
# from torch.utils.data import random_split
from pydantic.dataclasses import dataclass


# Define Generator
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


@dataclass
class ModelParameters:
    beta1: float
    nz: int


class GAN(pl.LightningModule):
    def __init__(self, config: ModelParameters):
        super(GAN, self).__init__()
        self.config = config
        self.generator = Generator(self.nz)
        self.discriminator = Discriminator()
        self.criterion = nn.BCELoss()

    def forward(self, z):
        return self.generator(z)

    def configure_optimizers(self):
        optimizerD = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.lr,
            betas=(self.config.beta1, 0.999),
        )
        optimizerG = optim.Adam(
            self.generator.parameters(),
            lr=self.config.lr,
            betas=(self.config.beta1, 0.999),
        )
        return [optimizerD, optimizerG], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_images = batch
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # Train Discriminator
        if optimizer_idx == 0:
            real_output = self.discriminator(real_images).view(-1, 1)
            d_loss_real = self.criterion(real_output, real_labels)
            z = torch.randn(batch_size, self.config.nz, 1, 1, device=self.device)
            fake_images = self.generator(z)
            fake_output = self.discriminator(fake_images.detach()).view(-1, 1)
            d_loss_fake = self.criterion(fake_output, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

        # Train Generator
        if optimizer_idx == 1:
            z = torch.randn(batch_size, self.config.nz, 1, 1, device=self.device)
            fake_images = self.generator(z)
            fake_output = self.discriminator(fake_images).view(-1, 1)
            g_loss = self.criterion(fake_output, real_labels)
            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss
