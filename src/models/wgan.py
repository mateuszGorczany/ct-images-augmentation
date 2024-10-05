import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from pydantic.dataclasses import dataclass
from src.utils import dicom_batch_to_video_batch, log_visualizations, log_common_metrics, video_batch_to_dicom_batch

import numpy as np
import torch
import os
from torch import log_, nn
from torch import optim
from torch.nn import functional as F
from typing import Literal

class Discriminator(nn.Module):
    def __init__(self, channel=512):
        super(Discriminator, self).__init__()        
        self.channel = channel
        n_class = 1
        
        self.conv1 = nn.Conv3d(1, channel//8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel//4)
        self.conv3 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel//2)
        self.conv4 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(channel)
        
        self.conv5 = nn.Conv3d(channel, n_class, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor, _return_activations=False):
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h5 = self.conv5(h4)
        output = h5

        return output
    

class Generator(nn.Module):
    def __init__(self, noise:int=1000, channel:int=64):
        super(Generator, self).__init__()
        _c = channel
        
        self.noise = noise
        self.fc = nn.Linear(1000,512*4*4*4)
        self.bn1 = nn.BatchNorm3d(_c*8)
        
        self.tp_conv2 = nn.Conv3d(_c*8, _c*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(_c*4)
        
        self.tp_conv3 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(_c*2)
        
        self.tp_conv4 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(_c)
        
        self.tp_conv5 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, noise: torch.Tensor):
        noise = noise.view(-1, 1000)
        h = self.fc(noise)
        h = h.view(-1,512,4,4,4)
        h = F.relu(self.bn1(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv2(h)
        h = F.relu(self.bn2(h))
        
        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv3(h)
        h = F.relu(self.bn3(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv4(h)
        h = F.relu(self.bn4(h))

        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv5(h)

        h = F.tanh(h)

        return h

@dataclass
class ModelParameters:
    learning_rate: float
    beta1: float
    beta2: float
    scan_depth: int
    nz: int
    p_lambda: float = 10.0


class GAN(pl.LightningModule):
    def __init__(self, config: ModelParameters):
        super(GAN, self).__init__()
        self.config = config
        self.generator: Generator = Generator(noise=config.nz, channel=config.scan_depth)
        self.discriminator: Discriminator = Discriminator()
        self.automatic_optimization = False

        # self.scaler_g = torch.cuda.amp.GradScaler()
        # self.scaler_d = torch.cuda.amp.GradScaler()

        self.save_hyperparameters()
        # self.validation_reconstructions = []


    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list]:
        optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )
        optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )
        return [optimizer_D, optimizer_G], []
    
    def discriminator_step(self, real_images, stage: Literal["train", "val"]):
        _batch_size = real_images.size(0)
        y_real_pred = self.discriminator.forward(real_images)

        d_real_loss = y_real_pred.mean()
        self.log(f"{stage}/d_real_loss", d_real_loss,  on_step=True, on_epoch=True, sync_dist=True)
        
        noise = torch.randn((_batch_size, self.config.nz, 1, 1, 1), device=self.device)
        fake_images = self.generator.forward(noise)
        y_fake_pred = self.discriminator.forward(fake_images.detach())

        d_fake_loss = y_fake_pred.mean()
        self.log(f"{stage}/d_fake_loss", d_fake_loss,  on_step=True, on_epoch=True, sync_dist=True)

        gradient_penalty = calc_gradient_penalty(self, real_images, fake_images)
    
        d_loss = - d_real_loss + d_fake_loss +gradient_penalty
        self.log(f"{stage}/d_loss", d_loss, on_step=True, on_epoch=True, sync_dist=True)
        Wasserstein_D = d_real_loss - d_fake_loss
        self.log(f"{stage}/d_wasserstein", Wasserstein_D,  on_step=True, on_epoch=True, sync_dist=True)

        return fake_images, d_loss
    
    def generator_step(self, real_images, stage: Literal["train", "val"]):
        _batch_size = real_images.size(0)
        noise = torch.randn((_batch_size, self.config.nz, 1, 1 ,1), device=self.device)
        fake_image = self.generator.forward(noise)
        y_fake_g = self.discriminator.forward(fake_image)
        g_loss = -y_fake_g.mean()
        self.log(f"{stage}/g_loss", g_loss,  on_step=True, on_epoch=True, sync_dist=True)

        return fake_image, g_loss


    def training_step(self, batch):
        real_images = dicom_batch_to_video_batch(batch)
        optimizer_d, optimizer_g = self.optimizers() # type: ignore

        # Discriminator
        if self.global_step % 5 == 0:
            self.toggle_optimizer(optimizer_d)
            optimizer_d.zero_grad()

            fake_images, d_loss =  self.discriminator_step(real_images, "train")
            self.manual_backward(d_loss)
            optimizer_d.step()
            # z = torch.randn(_batch_size, self.config.nz, 1, 1, device=self.device)
            # z = z.type_as(real_images)
            self.untoggle_optimizer(optimizer_d)

        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad()
        fake_image, g_loss = self.generator_step(real_images, "train")
        self.manual_backward(g_loss)
        optimizer_g.step()

        self.untoggle_optimizer(optimizer_g)

        log_common_metrics(self, real_images, fake_image, "train")
        # self.generated_imgs = self.generator(z
        # generated_imgs = self.generator(torch.randn_like(imgs))


        # self.log("epoch_loss", self.epoch_loss / (self.global_step + 1), prog_bar=True)
        # self.log(
        #     "generator_loss",
        #     self.generator_epoch_loss / (self.global_step + 1),
        #     prog_bar=True,
        # )
        # self.log(
        #     "discriminator_loss",
        #     self.discriminator_epoch_loss / (self.global_step),
        #     prog_bar=True,
        # )

    def validation_step(self, batch):
        # sample_imgs, _, _ = self(images)
        if not self.trainer.is_global_zero:
            return
        imgs = dicom_batch_to_video_batch(batch)

        fake_images, d_loss =  self.discriminator_step(imgs, "val")
        fake_images, g_loss = self.generator_step(imgs, "val")    

        log_common_metrics(
            self, imgs, fake_images, "val",
        )
        if self.current_epoch % 1 == 0:
            log_visualizations(
                self, imgs, fake_images
            )

def calc_gradient_penalty(gan: GAN, real_data, fake_data):    
    alpha = torch.rand(real_data.size(0),1,1,1,1, device=real_data.device)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.to(real_data.device)
    interpolates.requires_grad = True

    disc_interpolates = gan.discriminator(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates, 
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True
        )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gan.config.p_lambda
    return gradient_penalty