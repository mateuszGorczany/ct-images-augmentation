import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from pydantic.dataclasses import dataclass


class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.main = nn.Sequential()

    def forward(self, input):
        return self.main(input)


# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential()

    def forward(self, input):
        return self.main(input)


@dataclass
class ModelParameters:
    beta1: float
    nz: int
    learning_rate: float


class GAN(pl.LightningModule):
    def __init__(self, config: ModelParameters):
        super(GAN, self).__init__()
        self.config = config
        self.generator = Generator(self.config.nz)
        self.discriminator = Discriminator()
        self.criterion = nn.BCELoss()

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

    def training_step(self, batch, batch_idx):
        real_images = batch
        batch_size = real_images.size(0)

        optimizer_d, optimizer_g = self.optimizers()

        # Train Discriminator
        # real_output = self.discriminator(real_images).view(-1, 1)
        # d_loss_real = self.criterion(real_output, real_labels)
        # fake_images = self.generator(z)
        # fake_output = self.discriminator(fake_images.detach()).view(-1, 1)
        # d_loss_fake = self.criterion(fake_output, fake_labels)
        # d_loss = d_loss_real + d_loss_fake
        # self.log("d_loss", d_loss, prog_bar=True)

        # Train Generator
        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)
        z = torch.randn(batch_size, self.config.nz, 1, 1, device=self.device)
        z = z.type_as(real_images)
        self.generated_imgs = self.generator(z)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(real_images.size(0), 1)
        valid = valid.type_as(real_images)

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
        valid = torch.ones(real_images.size(0), 1)
        valid = valid.type_as(real_images)

        real_loss = self.adversarial_loss(self.discriminator(real_images), valid)

        # how well can it label as fake?
        fake = torch.zeros(real_images.size(0), 1)
        fake = fake.type_as(real_images)

        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
