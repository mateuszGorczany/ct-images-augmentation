import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from pydantic.dataclasses import dataclass
from src.utils import plot_dicom
from generative.networks.nets import AutoencoderKL, PatchDiscriminator
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.layers import Act
from monai.visualize import plot_2d_or_3d_image


class Autoencoder(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = AutoencoderKL(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_channels=(32, 64, 64),
            latent_channels=3,
            num_res_blocks=1,
            norm_num_groups=16,
            attention_levels=(False, False, True),
        )

    def forward(self, img):
        # img_flat = img.view(img.size(0), -1)
        # validity = self.model(img_flat)

        # return validity
        return self.model(img)

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, z):
        return self.model.decode(z)


# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.discriminator = PatchDiscriminator(
            spatial_dims=3,
            num_layers_d=3,
            num_channels=32,
            in_channels=1,
            out_channels=1,
            kernel_size=4,
            activation=(Act.LEAKYRELU, {"negative_slope": 0.2}),
            norm="BATCH",
            bias=False,
            padding=1,
        )

    def forward(self, input):
        return self.discriminator(input)


@dataclass
class ModelParameters:
    learning_rate: float
    img_shape: list
    beta1: float
    beta2: float
    adv_weight: float
    perceptual_weight: float
    kl_weight: float
    fake_3d_ratio: float
    device: str
    autoencoder_warm_up_n_epochs: int


class GAN(pl.LightningModule):
    def __init__(self, config: ModelParameters):
        super(GAN, self).__init__()
        self.save_hyperparameters()
        self.config = config
        self.autoencoder = Autoencoder(self.config.img_shape)
        self.discriminator = Discriminator(self.config.img_shape)
        # self.criterion = nn.BCELoss()
        self.automatic_optimization = False

        self.perceptual_loss = PerceptualLoss(
            spatial_dims=3,
            network_type="squeeze",
            fake_3d_ratio=self.config.fake_3d_ratio,
        )
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")

        self.scaler_g = torch.cuda.amp.GradScaler()
        self.scaler_d = torch.cuda.amp.GradScaler()

        self.epoch_loss = 0
        self.generator_epoch_loss = 0
        self.discriminator_epoch_loss = 0
        self.l1_loss = nn.L1Loss()

        self.validation_reconstructions = []

    def forward(self, z):
        return self.autoencoder(z)

    def configure_optimizers(self):
        optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )
        optimizer_G = optim.Adam(
            self.autoencoder.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )
        return [optimizer_D, optimizer_G], []

    def KL_loss(self, z_mu, z_sigma):
        kl_loss = 0.5 * torch.sum(
            z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
            dim=[1, 2, 3, 4],
        )
        return torch.sum(kl_loss) / kl_loss.shape[0]

    def training_step(self, batch):
        imgs = batch

        optimizer_d, optimizer_g = self.optimizers()

        # Autoencoder
        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad()
        reconstruction, z_mu, z_sigma = self.autoencoder(imgs)
        kl_loss = self.KL_loss(z_mu, z_sigma)

        recons_loss = self.l1_loss(reconstruction.float(), imgs.float())
        perceptual_loss = self.perceptual_loss(reconstruction.float(), imgs.float())
        g_loss = (
            recons_loss
            + (self.config.kl_weight * kl_loss)
            + (self.config.perceptual_weight * perceptual_loss)
        )

        if self.current_epoch > self.config.autoencoder_warm_up_n_epochs:
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = self.adv_loss(
                logits_fake, target_is_real=True, for_discriminator=False
            )
            g_loss += self.config.adv_weight * generator_loss

        self.manual_backward(g_loss)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        # Discriminator
        if self.current_epoch > self.config.autoencoder_warm_up_n_epochs:
            self.toggle_optimizer(optimizer_d)
            optimizer_d.zero_grad(set_to_none=True)
            logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = self.adv_loss(
                logits_fake, target_is_real=False, for_discriminator=True
            )
            logits_real = self.discriminator(imgs.contiguous().detach())[-1]
            loss_d_real = self.adv_loss(
                logits_real, target_is_real=True, for_discriminator=True
            )
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            d_loss = self.config.adv_weight * discriminator_loss

            self.manual_backward(d_loss)
            optimizer_d.step()

            self.untoggle_optimizer(optimizer_d)

        self.epoch_loss += recons_loss.item()
        if self.current_epoch > self.config.autoencoder_warm_up_n_epochs:
            self.generator_epoch_loss += generator_loss.item()
            self.discriminator_epoch_loss += discriminator_loss.item()

        self.log("epoch_loss", self.epoch_loss / (self.global_step + 1), prog_bar=True)
        self.log(
            "generator_loss",
            self.generator_epoch_loss / (self.global_step + 1),
            prog_bar=True,
        )
        self.log(
            "discriminator_loss",
            self.discriminator_epoch_loss / (self.global_step),
            prog_bar=True,
        )

        # Train generator
        # generate images
        # reconstruction, z_mu, z_sigma = self.generator(imgs)
        # logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]

        # recons_loss = F.l1_loss(reconstruction.float(), imgs.float())
        # p_loss = self.perceptual_loss(reconstruction.float(), imgs.float())
        # generator_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

        # kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        # g_loss = recons_loss + (self.config.kl_weight * kl_loss) + (self.config.perceptual_weight * p_loss) + (self.config.adv_weight * generator_loss)

        # self.log("g_loss", g_loss, prog_bar=True)
        # self.manual_backward(g_loss)
        # optimizer_g.step()
        # optimizer_g.zero_grad()
        # self.untoggle_optimizer(optimizer_g)

        # # Train discriminator
        # # Measure discriminator's ability to classify real from generated samples
        # self.toggle_optimizer(optimizer_d)

        # logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
        # loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        # logits_real = self.discriminator(imgs.contiguous().detach())[-1]
        # loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        # discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        # d_loss = self.config.adv_weight * discriminator_loss

        # self.log("d_loss", d_loss, prog_bar=True)
        # self.manual_backward(d_loss)
        # optimizer_d.step()
        # optimizer_d.zero_grad()
        # self.untoggle_optimizer(optimizer_d)

    def test_step(self, batch):
        if not self.trainer.is_global_zero:
            return
        sample_imgs = self.autoencoder(batch)
        grid = plot_dicom(sample_imgs[0], title="Reconstructed Test Images")
        self.logger.experiment.add_figure(
            "Regenerated scan by autoencoder", grid, self.current_epoch
        )

    def on_test_epoch_end(self):
        pass

    def validation_step(self, batch):
        # sample_imgs, _, _ = self(images)
        if not self.trainer.is_global_zero:
            return
        imgs = batch
        reconstruction, z_mu, z_sigma = self.autoencoder(imgs)
        self.validation_reconstructions.append(reconstruction)
        kl_loss = self.KL_loss(z_mu, z_sigma)

        recons_loss = self.l1_loss(reconstruction.float(), imgs.float())
        perceptual_loss = self.perceptual_loss(reconstruction.float(), imgs.float())
        g_loss = (
            recons_loss
            + (self.config.kl_weight * kl_loss)
            + (self.config.perceptual_weight * perceptual_loss)
        )

        self.log("validation_generator_loss", g_loss)

    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        plot_2d_or_3d_image(
            self.validation_reconstructions,
            step=self.global_step,
            writer=self.logger.experiment,
            frame_dim=-1,
            tag="validation_autoencoder_reconstruction",
        )

        self.validation_reconstructions = []

    def predict_step(self, batch):
        """Batch should be random noise"""
        if not self.trainer.is_global_zero:
            return
        random_noise = torch.randn_like(self.autoencoder.encode(batch)[0])
        generated_image = self.autoencoder.decode(random_noise)
        plot_2d_or_3d_image(
            generated_image,
            step=self.global_step + 1,
            writer=self.logger.experiment,
            frame_dim=-1,
            tag="Generated_scan",
        )

        grid = plot_dicom(generated_image[0], title="Generated image from random noise")
        self.logger.experiment.add_figure(
            "Generated image from random noise", grid, self.current_epoch
        )
