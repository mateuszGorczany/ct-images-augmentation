import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision
import numpy as np

from pydantic.dataclasses import dataclass
from src.utils import plot_dicom
import wandb


class Generator(nn.Module):
        
    def __init__(self, img_shape: tuple, latent_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.img_shape = img_shape
        self.latent_dim = latent_dim # aka z

        self.model = nn.Sequential(
            *self._block(self.latent_dim, 128, normalize=False),
            # *self._block(128, 256),
            # *self._block(256, 512),
            *self._block(128, 256),
            nn.Linear(256, int(np.prod(img_shape))),
            nn.Sigmoid(),
        )

    def _block(self, in_feat: int, out_feat: int, normalize=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
    # def __init__(self, nz):
    #     super(Generator, self).__init__()
    #     self.main = nn.Sequential(
    #         nn.ConvTranspose2d(nz, 64, 4, 1, 0, bias=False),
    #         nn.BatchNorm2d(64),
    #         nn.ReLU(True),
    #         nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
    #         nn.Tanh(),
    #     )

    # def forward(self, input):
    #     return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

# # Define Discriminator
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(1, 64, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 1, 4, 2, 1, bias=False),
#             nn.Sigmoid(),
#         )

#     def forward(self, input):
#         return self.main(input)


@dataclass
class ModelParameters:
    learning_rate: float
    img_shape: list
    latent_dim: int
    beta1: float
    beta2: float
    device: str


class GAN(pl.LightningModule):
    def __init__(self, config: ModelParameters):
        super(GAN, self).__init__()
        self.save_hyperparameters()
        self.config = config
        self.generator = Generator(self.config.img_shape, self.config.latent_dim)
        self.discriminator = Discriminator(self.config.img_shape)
        self.criterion = nn.BCELoss()
        self.automatic_optimization = False

        self.validation_z = torch.randn(8, self.config.latent_dim, device=config.device)
        self.example_input_array = torch.zeros(2, self.config.latent_dim)


    def forward(self, z):
        return self.generator(z)

    def configure_optimizers(self):
        optimizerD = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, 0.999),
        )
        optimizerG = optim.Adam(
            self.generator.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, 0.999),
        )
        return [optimizerD, optimizerG], []

    def adversarial_loss(self, y_hat, y):
        return torch.nn.functional.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch):
        # print(len(batch))
        imgs = batch
        # imgs = imgs['data']
        # imgs = batch

        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(imgs.shape[0], self.config.latent_dim)
        z = z.type_as(imgs)

        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(z)

        # log sampled images
        # sample_imgs = self.generated_imgs[:6]
        # Assuming sample_imgs has shape [6, 1, 256, 256, 32] as mentioned
        # First, permute to move the last dimension (channels) to the second position
        # sample_imgs = sample_imgs.squeeze().permute(0, 3, 1, 2)
        # Now, sample_imgs has shape [6, 32, 256, 256], which is (B x C x H x W)
        # grid = torchvision.utils.make_grid(sample_imgs)
        # grid = plot_dicom(sample_imgs[0].to("cpu"), title="Generated Images")
        # self.logger.log_image(key="generated_images", images=[grid,], step=self.global_step)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        # adversarial loss is binary cross-entropy
        g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)

        real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        # how well can it label as fake?
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)

        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def configure_optimizers(self):
        lr = self.config.learning_rate
        b1 = self.config.beta1
        b2 = self.config.beta2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        # sample_imgs = sample_imgs.squeeze().permute(0, 3, 1, 2)
        # grid = torchvision.utils.make_grid(sample_imgs)
        grid = plot_dicom(sample_imgs[0], title="Generated Images")
        # self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
        self.logger.log_image("generated_images", [grid,], self.current_epoch)

    # def training_step(self, batch, batch_idx):
    #     real_images = batch
    #     batch_size = real_images.size(0)
    #     batch_size = real_images.size(0)
    #     real_labels = torch.ones(batch_size, 1, device=self.device)
    #     fake_labels = torch.zeros(batch_size, 1, device=self.device)

    #     optimizer_d, optimizer_g = self.optimizers()

    #     # Train Discriminator
    #     self.toggle_optimizer(optimizer_d)
    #     real_output = self.discriminator(real_images).view(-1, 1)
    #     d_loss_real = self.criterion(real_output, real_labels)

    #     z = torch.randn(batch_size, self.config.nz, 1, 1, device=self.device)
    #     fake_images = self.generator(z)
    #     fake_output = self.discriminator(fake_images.detach()).view(-1, 1)
    #     d_loss_fake = self.criterion(fake_output, fake_labels)

    #     d_loss = d_loss_real + d_loss_fake
    #     self.log("d_loss", d_loss, prog_bar=True)
    #     self.manual_backward(d_loss)
    #     optimizer_d.step()
    #     optimizer_d.zero_grad()
    #     self.untoggle_optimizer(optimizer_d)
        

    #     # real_output = self.discriminator(real_images).view(-1, 1)
    #     # d_loss_real = self.criterion(real_output, real_labels)
    #     # fake_images = self.generator(z)
    #     # fake_output = self.discriminator(fake_images.detach()).view(-1, 1)
    #     # d_loss_fake = self.criterion(fake_output, fake_labels)
    #     # d_loss = d_loss_real + d_loss_fake
    #     # self.log("d_loss", d_loss, prog_bar=True)

    #     # Train Generator
    #     # train generator
    #     # generate images
    #     self.toggle_optimizer(optimizer_g)
    #     z = torch.randn(batch_size, self.config.nz, 1, 1, device=self.device)
    #     fake_images = self.generator(z)
    #     fake_output = self.discriminator(fake_images).view(-1, 1)
    #     g_loss = self.criterion(fake_output, real_labels)

    #     sample_imgs = self.generated_imgs[:6]
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     self.logger.experiment.add_image("generated_images", grid, 0)
    #     self.log("g_loss", g_loss, prog_bar=True)
    #     self.manual_backward(g_loss)
    #     optimizer_g.step()
    #     optimizer_g.zero_grad()
    #     self.untoggle_optimizer(optimizer_g)