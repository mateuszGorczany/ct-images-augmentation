import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from pydantic.dataclasses import dataclass
from src.utils import plot_dicom
from generative.networks.nets import DiffusionModelUNet
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
from src.models.monai_autoencoder import GAN
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from generative.inferers import LatentDiffusionInferer
from generative.networks.schedulers import DDPMScheduler
from pathlib import Path

@dataclass
class DiffusionModelParameters:
    spatial_dims: int
    in_channels: int
    out_channels: int
    num_res_blocks: int
    num_channels: list[int]
    attention_levels: list[bool]
    num_head_channels: list[int]

class DiffusionModel(nn.Module):

    def __init__(self, model_parameters: DiffusionModelParameters):
        super().__init__()

        self.unet = DiffusionModelUNet(
            spatial_dims=model_parameters.spatial_dims,
            in_channels=model_parameters.in_channels,
            out_channels=model_parameters.out_channels,
            num_res_blocks=model_parameters.num_res_blocks,
            num_channels=model_parameters.num_channels,
            attention_levels=model_parameters.attention_levels,
            num_head_channels=model_parameters.num_head_channels,
        )

class Autoencoder(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.autoencoder: GAN = GAN.load_from_checkpoint(checkpoint_path)

    def forward(self, img):
        return self.autoencoder(img)

    def encode(self, x):
        return self.autoencoder.autoencoder.model.encode(x)

    def decode(self, z):
        return self.autoencoder.autoencoder.model.decode(z)
    
    def encode_stage_2_inputs(self, x):
        return self.autoencoder.autoencoder.model.encode_stage_2_inputs(x)


@dataclass
class SchedulerParameters:
    num_train_steps: int
    beta_start: float
    beta_end: float
    schedule: str

@dataclass
class ModelParameters:
    scheduler_parameters: SchedulerParameters
    diffusion_model_parameters: DiffusionModelParameters
    autoencoder_checkpoint_path: str
    learning_rate: float

    img_shape: list
    beta1: float
    beta2: float
    scale_factor: float


class Model(pl.LightningModule):
    def __init__(self, config: ModelParameters):
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.config: ModelParameters = config
        self.autoencoder = Autoencoder(config.autoencoder_checkpoint_path)

        self.scaler = GradScaler()
        self.mu = torch.load(Path(config.autoencoder_checkpoint_path).parent.parent/"mu_mean.pt", map_location=self.device).to(self.device)
        self.sigma = torch.load(Path(config.autoencoder_checkpoint_path).parent.parent/"sigma_mean.pt", map_location=self.device).to(self.device)

        self.diffusion_model = DiffusionModel(config.diffusion_model_parameters)

        self.scheduler = DDPMScheduler(
            num_train_timesteps=config.scheduler_parameters.num_train_steps, 
            schedule=config.scheduler_parameters.schedule,
            beta_start=config.scheduler_parameters.beta_start,
            beta_end=config.scheduler_parameters.beta_end
        )
        self.inferer = LatentDiffusionInferer(
            scheduler=self.scheduler,
            scale_factor=config.scale_factor,

        )

    def forward(self, z):
        return self.autoencoder(z)

    def configure_optimizers(self):
        optimizer_diff = optim.Adam(
            self.diffusion_model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )
        return [optimizer_diff], []

    def loss(self, noise, noise_pred):
        return F.mse_loss(noise_pred.float(), noise.float())

    def _step(self, batch):
        images = batch

        # noise = torch.randn_like(self.autoencoder.encode_stage_2_inputs(images)).to(self.device)
        noise = self.__sample_noise_z().to(self.device)

        timesteps = torch.randint(
            0, 
            self.inferer.scheduler.num_train_timesteps,
            (images.shape[0],),
            device=self.device
        ).long()

        noise_pred = self.inferer(
            inputs=images, 
            autoencoder_model=self.autoencoder.autoencoder.autoencoder.model, 
            diffusion_model=self.diffusion_model.unet, 
            noise=noise, 
            timesteps=timesteps
        )

        return self.loss(noise_pred.float(), noise.float())

    def training_step(self, batch):
        images = batch

        optimizer_d = self.optimizers()

        loss = self._step(images)

        self.log("step_loss", loss / (self.global_step + 1), prog_bar=True)

    def test_step(self, batch):
        if not self.trainer.is_global_zero:
            return
        
        loss = self._step(batch)
        grid = plot_dicom(self.__synthetic_image(), title="Reconstructed Test Images")
        self.logger.experiment.add_figure(
            f"Generated scan by diffusion, loss ={loss}", grid, self.current_epoch
        )

    def on_test_epoch_end(self):
        pass

    def validation_step(self, batch):
        # sample_imgs, _, _ = self(images)
        if not self.trainer.is_global_zero:
            return
        imgs = batch

        val_loss = self._step(imgs)
        self.log("validation_loss", val_loss)

    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        
        synthetic_image = self.__synthetic_image()
        print(synthetic_image.shape)
        plot_2d_or_3d_image(
            synthetic_image[0],
            step=self.global_step,
            writer=self.logger.experiment,
            frame_dim=-1,
            tag="validation_generation",
        )
        if self.current_epoch % 50 == 0:
            plot_2d_or_3d_image(
                self.__synthetic_image(),
                step=self.global_step,
                writer=self.logger.experiment,
                frame_dim=-1,
                tag="validation_generation_3d",
            )
            
    def __sample_noise_z(self) -> torch.Tensor:
        eps = torch.randn_like(self.mu)
        return self.mu + eps*self.sigma
    
    def __synthetic_image(self) -> np.ndarray:
        # noise = torch.randn((1, 3, 24, 24, 16)).to(self.device)
        noise = self.__sample_noise_z().to(self.device)
        self.scheduler.set_timesteps(num_inference_steps=1000)
        synthetic_images = self.inferer.sample(
            input_noise=noise, autoencoder_model=self.autoencoder.autoencoder.autoencoder.model, diffusion_model=self.diffusion_model.unet, scheduler=self.scheduler
        )
        synthetic_image = synthetic_images[0].detach().cpu().numpy()

        return synthetic_image

    def predict_step(self, batch):
        """Batch should be random noise"""
        if not self.trainer.is_global_zero:
            return
        synthetic_image = self.__synthetic_image()
        plot_2d_or_3d_image(
            synthetic_image,
            step=self.global_step + 1,
            writer=self.logger.experiment,
            frame_dim=-1,
            tag="Generated_scan",
        )

        grid = plot_dicom(synthetic_image, title="Generated image from random noise")
        self.logger.experiment.add_figure(
            "Generated image from random noise", grid, self.current_epoch
        )
