"""Adapted from https://github.com/SongweiGe/TATS"""
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import pytorch_lightning as pl
from pydantic.dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.medical_diffusion.vq_gan_3d.model.vqgan import (
    Encoder,
    Decoder,
    NLayerDiscriminator,
    NLayerDiscriminator3D,
    shift_dim,
    adopt_weight,
    SamePadConv3d,
    vanilla_d_loss,
    hinge_d_loss,
)
from src.models.medical_diffusion.vq_gan_3d.model.lpips import LPIPS
from src.models.medical_diffusion.vq_gan_3d.model.codebook import Codebook
from src.utils import log_common_metrics, log_visualizations, plot_dicom, log_comparison
from src.utils import plot_dicom
from monai.visualize.img2tensorboard import plot_2d_or_3d_image


# class GAN(pl.LightningModule):
#     def __init__(self, config: ModelParameters):
#         super(GAN, self).__init__()
#         self.save_hyperparameters()
#         self.config = config
#         self.autoencoder = Autoencoder(self.config.img_shape)
#         self.discriminator = Discriminator(self.config.img_shape)
#         # self.criterion = nn.BCELoss()
#         self.automatic_optimization = False

#         self.perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", fake_3d_ratio=self.config.fake_3d_ratio)
#         self.adv_loss = PatchAdversarialLoss(criterion="least_squares")

#         self.scaler_g = torch.cuda.amp.GradScaler()
#         self.scaler_d = torch.cuda.amp.GradScaler()

#         self.epoch_loss = 0
#         self.generator_epoch_loss = 0
#         self.discriminator_epoch_loss = 0
#         self.l1_loss = nn.L1Loss()

#         self.validation_reconstructions = []

#     def forward(self, z):
#         return self.autoencoder(z)

#     def configure_optimizers(self):
#         optimizer_D = optim.Adam(
#             self.discriminator.parameters(),
#             lr=self.config.learning_rate,
#             betas=(self.config.beta1, self.config.beta2),
#         )
#         optimizer_G = optim.Adam(
#             self.autoencoder.parameters(),
#             lr=self.config.learning_rate,
#             betas=(self.config.beta1, self.config.beta2),
#         )
#         return [optimizer_D, optimizer_G], []


#     def KL_loss(self, z_mu, z_sigma):
#         kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
#         return torch.sum(kl_loss) / kl_loss.shape[0]


#     def training_step(self, batch):
#         imgs = batch

#         optimizer_d, optimizer_g = self.optimizers()

#         # Generator
#         self.toggle_optimizer(optimizer_g)
#         optimizer_g.zero_grad()
#         reconstruction, z_mu, z_sigma = self.autoencoder(imgs)
#         kl_loss = self.KL_loss(z_mu, z_sigma)

#         recons_loss = self.l1_loss(reconstruction.float(), imgs.float())
#         perceptual_loss = self.perceptual_loss(reconstruction.float(), imgs.float())
#         g_loss = recons_loss \
#             + (self.config.kl_weight * kl_loss) \
#             + (self.config.perceptual_weight * perceptual_loss)


#         if self.current_epoch > self.config.autoencoder_warm_up_n_epochs:
#             logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
#             generator_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
#             g_loss += self.config.adv_weight * generator_loss

#         self.manual_backward(g_loss)
#         optimizer_g.step()
#         self.untoggle_optimizer(optimizer_g)


#         # Autoencoder
#         if self.current_epoch > self.config.autoencoder_warm_up_n_epochs:
#             self.toggle_optimizer(optimizer_d)
#             optimizer_d.zero_grad(set_to_none=True)
#             logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
#             loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
#             logits_real = self.discriminator(imgs.contiguous().detach())[-1]
#             loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
#             discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

#             d_loss = self.config.adv_weight * discriminator_loss

#             self.manual_backward(d_loss)
#             optimizer_d.step()

#             self.untoggle_optimizer(optimizer_d)

#         self.epoch_loss += recons_loss.item()
#         if self.current_epoch > self.config.autoencoder_warm_up_n_epochs:
#             self.generator_epoch_loss += generator_loss.item()
#             self.discriminator_epoch_loss += discriminator_loss.item()

#         self.log("epoch_loss", self.epoch_loss / (self.global_step+1), prog_bar=True)
#         self.log("generator_loss", self.generator_epoch_loss / (self.global_step + 1), prog_bar=True)
#         self.log("discriminator_loss", self.discriminator_epoch_loss / (self.global_step), prog_bar=True)


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

# def test_step(self, batch):
#     if not self.trainer.is_global_zero:
#         return
#     sample_imgs = self.autoencoder(batch)
#     grid = plot_dicom(sample_imgs[0], title="Reconstructed Test Images")
#     self.logger.experiment.add_figure("Regenerated scan by autoencoder", grid, self.current_epoch)

# def on_test_epoch_end(self):
#     pass

# def validation_step(self, batch):
#     # sample_imgs, _, _ = self(images)
#     if not self.trainer.is_global_zero:
#         return
#     imgs = batch
#     reconstruction, z_mu, z_sigma = self.autoencoder(imgs)
#     self.validation_reconstructions.append(reconstruction)
#     kl_loss = self.KL_loss(z_mu, z_sigma)

#     recons_loss = self.l1_loss(reconstruction.float(), imgs.float())
#     perceptual_loss = self.perceptual_loss(reconstruction.float(), imgs.float())
#     g_loss = recons_loss \
#         + (self.config.kl_weight * kl_loss) \
#         + (self.config.perceptual_weight * perceptual_loss)

#     self.log("validation_generator_loss", g_loss)

# def on_validation_epoch_end(self):
#     if not self.trainer.is_global_zero:
#         return
#     plot_2d_or_3d_image(
#         self.validation_reconstructions,
#         step=self.global_step,
#         writer=self.logger.experiment,
#         frame_dim=-1,
#         tag='validation_autoencoder_reconstruction'
#     )

#     self.validation_reconstructions = []

# def predict_step(self, batch):
#     """Batch should be random noise"""
#     if not self.trainer.is_global_zero:
#         return
#     random_noise = torch.randn_like(self.autoencoder.encode(batch)[0])
#     generated_image =  self.autoencoder.decode(random_noise)
#     plot_2d_or_3d_image(
#         generated_image,
#         step=self.global_step+1,
#         writer=self.logger.experiment,
#         frame_dim=-1,
#         tag='Generated_scan'
#     )


#     grid = plot_dicom(generated_image[0], title="Generated image from random noise")
#     self.logger.experiment.add_figure("Generated image from random noise", grid, self.current_epoch)
@dataclass
class ModelParameters:
    embedding_dim: int
    n_codes: int
    n_hiddens: int
    downsample: list
    image_channels: int
    norm_type: str
    padding_type: str
    num_groups: int
    sample_every_n_epochs: int = 50


class VQGAN(pl.LightningModule):
    def __init__(self, config: ModelParameters):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.n_codes = config.n_codes

        self.encoder = Encoder(
            config.n_hiddens,
            config.downsample,
            config.dataset.image_channels,
            config.norm_type,
            config.padding_type,
            config.num_groups,
        )
        self.decoder = Decoder(
            config.n_hiddens,
            config.downsample,
            config.dataset.image_channels,
            config.norm_type,
            config.num_groups,
        )
        self.enc_out_ch = self.encoder.out_channels
        self.pre_vq_conv = SamePadConv3d(
            self.enc_out_ch, config.embedding_dim, 1, padding_type=config.padding_type
        )
        self.post_vq_conv = SamePadConv3d(config.embedding_dim, self.enc_out_ch, 1)

        self.codebook = Codebook(
            config.n_codes,
            config.embedding_dim,
            no_random_restart=config.no_random_restart,
            restart_thres=config.restart_thres,
        )

        self.gan_feat_weight = config.gan_feat_weight
        # TODO: Changed batchnorm from sync to normal
        self.image_discriminator = NLayerDiscriminator(
            config.dataset.image_channels,
            config.disc_channels,
            config.disc_layers,
            norm_layer=nn.BatchNorm2d,
        )
        self.video_discriminator = NLayerDiscriminator3D(
            config.dataset.image_channels,
            config.disc_channels,
            config.disc_layers,
            norm_layer=nn.BatchNorm3d,
        )

        if config.disc_loss_type == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif config.disc_loss_type == "hinge":
            self.disc_loss = hinge_d_loss

        self.perceptual_model = LPIPS().eval()

        self.image_gan_weight = config.image_gan_weight
        self.video_gan_weight = config.video_gan_weight

        self.perceptual_weight = config.perceptual_weight

        self.l1_weight = config.l1_weight
        self.save_hyperparameters()

    def encode(self, x, include_embeddings=False, quantize=True):
        h = self.pre_vq_conv(self.encoder(x))
        if quantize:
            vq_output = self.codebook(h)
            if include_embeddings:
                return vq_output["embeddings"], vq_output["encodings"]
            else:
                return vq_output["encodings"]
        return h

    def decode(self, latent, quantize=False):
        if quantize:
            vq_output = self.codebook(latent)
            latent = vq_output["encodings"]
        h = F.embedding(latent, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        return self.decoder(h)

    def forward(self, x, optimizer_idx=None, log_image=False):
        z = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output["embeddings"]))

        return z, vq_output, x_recon

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = batch
        # From (B, C, H, W, T) to (B, C, T, H, W)
        x = x.permute(0, 1, 4, 2, 3)
        B, C, T, H, W = x.shape

        z, vq_output, x_recon = self.forward(x)

        optimizer_ae, optimizer_d = self.optimizers()

        recon_loss = F.l1_loss(x_recon, x) * self.l1_weight

        # Selects one random 2D image from each 3D Image
        frame_idx = torch.randint(0, T, [B]).cuda()
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        # if log_image:
        #     return frames, frames_recon, x, x_recon

        # if optimizer_idx == 0:
        self.toggle_optimizer(optimizer_ae)
        # Autoencoder - train the "generator"

        # Perceptual loss
        perceptual_loss = 0
        if self.perceptual_weight > 0:
            perceptual_loss = (
                self.perceptual_model(frames, frames_recon).mean()
                * self.perceptual_weight
            )

        # Discriminator loss (turned on after a certain epoch)
        logits_image_fake, pred_image_fake = self.image_discriminator(frames_recon)
        logits_video_fake, pred_video_fake = self.video_discriminator(x_recon)
        g_image_loss = -torch.mean(logits_image_fake)
        g_video_loss = -torch.mean(logits_video_fake)
        g_loss = (
            self.image_gan_weight * g_image_loss + self.video_gan_weight * g_video_loss
        )
        disc_factor = adopt_weight(
            self.global_step, threshold=self.config.discriminator_iter_start
        )
        aeloss = disc_factor * g_loss

        # GAN feature matching loss - tune features such that we get the same prediction result on the discriminator
        image_gan_feat_loss = 0
        video_gan_feat_loss = 0
        feat_weights = 4.0 / (3 + 1)
        if self.image_gan_weight > 0:
            logits_image_real, pred_image_real = self.image_discriminator(frames)
            for i in range(len(pred_image_fake) - 1):
                image_gan_feat_loss += (
                    feat_weights
                    * F.l1_loss(pred_image_fake[i], pred_image_real[i].detach())
                    * (self.image_gan_weight > 0)
                )
        if self.video_gan_weight > 0:
            logits_video_real, pred_video_real = self.video_discriminator(x)
            for i in range(len(pred_video_fake) - 1):
                video_gan_feat_loss += (
                    feat_weights
                    * F.l1_loss(pred_video_fake[i], pred_video_real[i].detach())
                    * (self.video_gan_weight > 0)
                )
        gan_feat_loss = (
            disc_factor
            * self.gan_feat_weight
            * (image_gan_feat_loss + video_gan_feat_loss)
        )

        self.log(
            "train/g_image_loss",
            g_image_loss,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/g_video_loss",
            g_video_loss,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/image_gan_feat_loss",
            image_gan_feat_loss,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/video_gan_feat_loss",
            video_gan_feat_loss,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/perceptual_loss",
            perceptual_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/recon_loss",
            recon_loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/aeloss",
            aeloss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/commitment_loss",
            vq_output["commitment_loss"],
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/perplexity",
            vq_output["perplexity"],
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        # return (
        #     recon_loss,
        #     x_recon,
        #     vq_output,
        #     aeloss,
        #     perceptual_loss,
        #     gan_feat_loss,
        # )
        self.untoggle_optimizer(optimizer_ae)

        # if optimizer_idx == 1:
        # Train discriminator
        self.toggle_optimizer(optimizer_d)
        logits_image_real, _ = self.image_discriminator(frames.detach())
        logits_video_real, _ = self.video_discriminator(x.detach())

        logits_image_fake, _ = self.image_discriminator(frames_recon.detach())
        logits_video_fake, _ = self.video_discriminator(x_recon.detach())

        d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
        d_video_loss = self.disc_loss(logits_video_real, logits_video_fake)
        disc_factor = adopt_weight(
            self.global_step, threshold=self.config.discriminator_iter_start
        )
        discloss = disc_factor * (
            self.image_gan_weight * d_image_loss + self.video_gan_weight * d_video_loss
        )

        self.untoggle_optimizer(optimizer_d)

        self.log(
            "train/logits_image_real",
            logits_image_real.mean().detach(),
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/logits_image_fake",
            logits_image_fake.mean().detach(),
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/logits_video_real",
            logits_video_real.mean().detach(),
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/logits_video_fake",
            logits_video_fake.mean().detach(),
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/d_image_loss",
            d_image_loss,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/d_video_loss",
            d_video_loss,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/discloss",
            discloss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        # return discloss

        perceptual_loss = (
            self.perceptual_model(frames, frames_recon) * self.perceptual_weight
        )
        return recon_loss, x_recon, vq_output, perceptual_loss

        # if optimizer_idx == 0:
        #     (
        #         recon_loss,
        #         _,
        #         vq_output,
        #         aeloss,
        #         perceptual_loss,
        #         gan_feat_loss,
        #     ) = self.forward(x, optimizer_idx)
        # commitment_loss = vq_output["commitment_loss"]
        loss = recon_loss + commitment_loss + aeloss + perceptual_loss + gan_feat_loss
        # if optimizer_idx == 1:
        # discloss = self.forward(x, optimizer_idx)
        # loss = discloss
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        # From (B, C, H, W, T) to (B, C, T, H, W)
        x = x.permute(0, 1, 4, 2, 3)
        B, C, T, H, W = x.shape

        # recon_loss, _, vq_output, perceptual_loss = self.forward(x)
        z, vq_output, x_recon = self.forward(x)

        frame_idx = torch.randint(0, T, [B]).cuda()
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        perceptual_loss = (
            self.perceptual_model(frames, frames_recon) * self.perceptual_weight
        )

        recon_loss = F.l1_loss(x_recon, x) * self.l1_weight
        self.log("val/recon_loss", recon_loss, prog_bar=True)
        self.log("val/perceptual_loss", perceptual_loss, prog_bar=True)
        self.log("val/perplexity", vq_output["perplexity"], prog_bar=True)
        self.log("val/commitment_loss", vq_output["commitment_loss"], prog_bar=True)
        
        log_common_metrics(self, x, x_recon, "val")
        # ----------------- Save Image ------------------------------
        if self.current_epoch % self.sample_every_n_epochs == 0:
            log_visualizations(self, x, x_recon)

    def configure_optimizers(self):
        lr = self.config.lr
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.pre_vq_conv.parameters())
            + list(self.post_vq_conv.parameters())
            + list(self.codebook.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(
            list(self.image_discriminator.parameters())
            + list(self.video_discriminator.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        return [opt_ae, opt_disc], []

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch
        x = x.to(self.device)
        frames, frames_rec, _, _ = self(x, log_image=True)
        log["inputs"] = frames
        log["reconstructions"] = frames_rec
        # log['mean_org'] = batch['mean_org']
        # log['std_org'] = batch['std_org']
        return log

    def log_videos(self, batch, **kwargs):
        log = dict()
        x = batch
        _, _, x, x_rec = self(x, log_image=True)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        # log['mean_org'] = batch['mean_org']
        # log['std_org'] = batch['std_org']
        return log

    def predict_step(self, batch):
        """Batch should be random noise"""
        if not self.trainer.is_global_zero:
            return
        x = batch
        # From (B, C, H, W, T) to (B, C, T, H, W)
        x = x.permute(0, 1, 4, 2, 3)
        B, C, T, H, W = x.shape

        # recon_loss, _, vq_output, perceptual_loss = self.forward(x)
        z, vq_output, x_recon = self.forward(x)
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
