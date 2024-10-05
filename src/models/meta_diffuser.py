import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from pydantic.dataclasses import dataclass
from src.utils import log_common_metrics, log_visualizations, plot_dicom
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from generative.inferers import LatentDiffusionInferer
from generative.networks.schedulers import DDPMScheduler
from pathlib import Path
from src.models.medical_diffusion.ddpm.diffusion import GaussianDiffusion, EMA, Unet3D
import copy
from src.utils import dicom_batch_to_video_batch, video_batch_to_dicom_batch, log_common_metrics, log_visualizations

# @dataclass
# class ModelParameters:
#     scheduler_parameters: SchedulerParameters
#     diffusion_model_parameters: DiffusionModelParameters
#     autoencoder_checkpoint_path: str
#     learning_rate: float

#     img_shape: list
#     beta1: float
#     beta2: float
#     scale_factor: float


# class Trainer(object):
#     def __init__(
#         self,
#         config: ModelParameters,
#     ):
#         super().__init__()
        # self.model = diffusion_model
        # self.update_ema_every = config.update_ema_every

        # self.step_start_ema = config.step_start_ema
        # self.save_and_sample_every = config.save_and_sample_every

        # self.batch_size = confgi.train_batch_size
        # self.image_size = diffusion_model.image_size
        # self.gradient_accumulate_every = config.gradient_accumulate_every
        # self.train_num_steps = config.train_num_steps

        # image_size = diffusion_model.image_size
        # channels = diffusion_model.channels
        # num_frames = diffusion_model.num_frames

        # self.cfg = cfg
        # if dataset:
        #     self.ds = dataset
        # else:
        #     assert folder is not None, "Provide a folder path to the dataset"
        #     self.ds = Dataset(
        #         folder, image_size, channels=channels, num_frames=num_frames
        #     )

        # self.dl = cycle(dl)

        # print(f"found {len(self.ds)} videos as gif files at {folder}")
        # assert (
        #     len(self.ds) > 0
        # ), "need to have at least 1 video to start training (although 1 is not great, try 100k)"


        # self.step = 0

        # self.amp = amp
        # self.scaler = GradScaler(enabled=amp)
        # self.max_grad_norm = max_grad_norm

        # self.num_sample_rows = num_sample_rows
        # self.results_folder = Path(results_folder)
        # self.results_folder.mkdir(exist_ok=True, parents=True)

        # self.__reset_parameters()


    # def save(self, milestone):
    #     data = {
    #         "step": self.step,
    #         "model": self.model.state_dict(),
    #         "ema": self.ema_model.state_dict(),
    #         "scaler": self.scaler.state_dict(),
    #     }
    #     torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))

    # def load(self, milestone, map_location=None, **kwargs):
    #     if milestone == -1:
    #         all_milestones = [
    #             int(p.stem.split("-")[-1])
    #             for p in Path(self.results_folder).glob("**/*.pt")
    #         ]
    #         assert (
    #             len(all_milestones) > 0
    #         ), "need to have at least one milestone to load from latest checkpoint (milestone == -1)"
    #         milestone = max(all_milestones)

    #     if map_location:
    #         data = torch.load(milestone, map_location=map_location)
    #     else:
    #         data = torch.load(milestone)

    #     self.step = data["step"]
    #     self.model.load_state_dict(data["model"], **kwargs)
    #     self.ema_model.load_state_dict(data["ema"], **kwargs)
    #     self.scaler.load_state_dict(data["scaler"])

@dataclass
class SchedulerParameters:
    # num_train_steps: int
    timesteps: int

@dataclass
class UnetParameters:
    dim_mults: list[int]

@dataclass
class EmaParameters:
    decay: float = 0.995
    start_step: int = 2000
    update_every_step: int = 10

@dataclass
class ScanParameters:
    img_size: int
    depth_size: int
    num_channels: int

@dataclass
class ModelParameters:
    unet_parameters: UnetParameters
    ema_parameters: EmaParameters
    scan_parameters: ScanParameters
    scheduler_parameters: SchedulerParameters
    vqgan_checkpoint_path: str
    loss_type: str
    prob_focus_present: float
    grad_scaler_enabled: bool
    learning_rate: float
    focus_present_mask: list | None = None
    max_grad_norm: float | None = None
    # train_num_steps: int = 100000
    # save_and_sample_every: int = 1000
    # max_grad_norm: float = None

class Model(pl.LightningModule):
    def __init__(self, config: ModelParameters):

        super(Model, self).__init__()
        self.config: ModelParameters = config
        self.unet = Unet3D(
            dim=config.scan_parameters.img_size,
            dim_mults=config.unet_parameters.dim_mults,
            channels=config.scan_parameters.num_channels,
        )
        self.diffusion_model = GaussianDiffusion(
            self.unet,
            vqgan_ckpt=config.vqgan_checkpoint_path,
            image_size=config.scan_parameters.img_size,
            num_frames=config.scan_parameters.depth_size,
            channels=config.scan_parameters.num_channels,
            timesteps=config.scheduler_parameters.timesteps,
            # sampling_timesteps=config.sampling_timesteps,
            loss_type=config.loss_type,
            # objective=cfg.objective
        )
        self.ema = EMA(config.ema_parameters.decay)
        self.ema_model: GaussianDiffusion = copy.deepcopy(self.diffusion_model)
        # self.vqgan = VQGAN.load_from_checkpoint(config.vqgan_ckpt)
        self.scaler = GradScaler(enabled=self.config.grad_scaler_enabled)
        self.__reset_parameters() 

        self.save_hyperparameters()
        self.automatic_optimization = False
        self._should_log_val = True
        self._should_log_train = True


    # def forward(self, z):
    #     return self.autoencoder(z)

    def configure_optimizers(self):
        optimizer_diff = optim.Adam(
            self.diffusion_model.parameters(),
            lr=self.config.learning_rate,
            # betas=(self.config.beta1, self.config.beta2),
        )
        return [optimizer_diff], []

    def __reset_parameters(self):
        self.ema_model.load_state_dict(self.diffusion_model.state_dict())

    def __step_ema(self):
        if self.global_step < self.config.ema_parameters.start_step:
            self.__reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.diffusion_model)

    def _step(self, batch):
        data = dicom_batch_to_video_batch(batch)

        return self.diffusion_model(
            data,
            prob_focus_present=self.config.prob_focus_present,
            focus_present_mask=self.config.focus_present_mask,
        )
    

    def training_step(self, batch):
# self, prob_focus_present=0.0, focus_present_mask=None, log_fn=noop):
        opt = self.optimizers()
        loss = self._step(batch)
        self.log("train_loss", loss.item(), prog_bar=True, sync_dist=True)
        self.manual_backward(self.scaler.scale(loss))

        if self.config.max_grad_norm:
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(
                self.diffusion_model.parameters(), self.config.max_grad_norm
            )

        self.scaler.step(opt)
        self.scaler.update()
        opt.zero_grad()

        if self.global_step % self.config.ema_parameters.update_every_step == 0:
            self.__step_ema()
        
        if self._should_log_train:
            self.log_all(batch, "train")
        self._should_log_train = False

    def on_train_epoch_end(self):
        self._should_log_train = True
    
    def __synthetic_image(self, bathc_size=1):
        batch_vid_samples = self.diffusion_model.sample(None, batch_size=1)
        batch_dicom_samples = video_batch_to_dicom_batch(batch_vid_samples)
        return batch_dicom_samples[0].detach().cpu().numpy(),

        #         if self.step != 0 and self.step % self.save_and_sample_every == 0:
        #     self.ema_model.eval()

        #     with torch.no_grad():
        #         milestone = self.step // self.save_and_sample_every
        #         num_samples = self.num_sample_rows**2
        #         batches = num_to_groups(num_samples, self.batch_size)

        #         all_videos_list = list(
        #             map(lambda n: self.ema_model.sample(batch_size=n), batches)
        #         )
        #         all_videos_list = torch.cat(all_videos_list, dim=0)

        #     all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))

        #     one_gif = rearrange(
        #         all_videos_list,
        #         "(i j) c f h w -> c f (i h) (j w)",
        #         i=self.num_sample_rows,
        #     )
        #     video_path = str(self.results_folder / str(f"{milestone}.gif"))
        #     video_tensor_to_gif(one_gif, video_path)
        #     log = {**log, "sample": video_path}

        #     # Selects one random 2D image from each 3D Image
        #     B, C, D, H, W = all_videos_list.shape
        #     frame_idx = torch.randint(0, D, [B]).cuda()
        #     frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(
        #         1, C, 1, H, W
        #     )
        #     frames = torch.gather(all_videos_list, 2, frame_idx_selected).squeeze(2)

        #     path = str(self.results_folder / f"sample-{milestone}.jpg")
        #     plt.figure(figsize=(50, 50))
        #     cols = 5
        #     for num, frame in enumerate(frames.cpu()):
        #         plt.subplot(math.ceil(len(frames) / cols), cols, num + 1)
        #         plt.axis("off")
        #         plt.imshow(frame[0], cmap="gray")
        #         plt.savefig(path)

        #     self.save(milestone)

        # log_fn(log)
        # self.step += 1

    # print("training completed")

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
        self.log("val/loss", val_loss.item(), prog_bar=True, sync_dist=True)

        if self._should_log_val:
            self.log_all(imgs, "val")
        self._should_log_val = False

    def log_all(self, batch, stage):    
        imgs = batch[:2]
        if not self.trainer.is_global_zero:
            return
        if self.current_epoch % 50 != 0:
            return
        batch_vid_samples = self.diffusion_model.sample(None, batch_size=2)
        batch_vid_sampels = video_batch_to_dicom_batch(batch_vid_samples)
        log_common_metrics(self, imgs, batch_vid_sampels, stage)
        log_visualizations(self, x=imgs, x_out=batch_vid_sampels)

    def on_validation_epoch_end(self):
        self._should_log_val = True
        # plot_2d_or_3d_image(
        #     synthetic_image[0],
        #     step=self.global_step,
        #     writer=self.logger.experiment,
        #     frame_dim=-1,
        #     tag="validation_generation",
        # )
        # plot_2d_or_3d_image(
        #     self.__synthetic_image(),
        #     step=self.global_step,
        #     writer=self.logger.experiment,
        #     frame_dim=-1,
        #     tag="validation_generation_3d",
        # )

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
