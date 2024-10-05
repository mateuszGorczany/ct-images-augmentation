from math import log
import matplotlib.pyplot as plt
from monai.visualize import matshow3d
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.utils import make_grid
from pytorch_msssim import ssim
from typing import Literal
from monai.visualize.img2tensorboard import plot_2d_or_3d_image
import lpips


def plot_dicom(volume, title="Scan"):
    fig = plt.figure(figsize=(1, 32))
    matshow3d(volume=volume, fig=fig, title=title, every_n=1, frame_dim=-1, cmap="gray")

    return fig


def checkpoints_dir_path(model_name: str) -> Path:
    directory = Path(f"./models/{model_name}/checkpoints/")
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def tensor_to_nii(tensor: torch.Tensor, file_path):
    # Ensure the tensor is on the CPU and convert to NumPy array
    np_array = tensor.cpu().numpy()

    # Create a NIfTI image
    nifti_img = nib.Nifti1Image(np_array, affine=np.eye(4))

    # Save the NIfTI image to the specified file path
    nib.save(nifti_img, file_path)


class LogCheckpointPathCallback(pl.Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint_path = trainer.checkpoint_callback.best_model_path
        trainer.logger.experiment.add_text("checkpoint_path", checkpoint_path)
        print(f"Checkpoint saved at: {checkpoint_path}")

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint_path = trainer.logger.experiment.get_text("checkpoint_path")
        print(f"Loading checkpoint from: {checkpoint_path}")


class WandbModelCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        checkpoint_path = self.best_model_path
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(checkpoint_path, name="best_model.ckpt")
        wandb.log_artifact(artifact)
        wandb.log({"checkpoint_path": checkpoint_path})
        print(f"Checkpoint saved to wandb: {checkpoint_path}")


def dicom_batch_to_video_batch(dicom: torch.Tensor) -> torch.Tensor:
    # (b, c, h, w, t) -> (b, c, t, h, w)
    return dicom.permute(0, 1, 4, 2, 3)


def video_batch_to_dicom_batch(video: torch.Tensor) -> torch.Tensor:
    # (b, c, t, h, w) -> (b, c, h, w, t)
    return video.permute(0, 1, 3, 4, 2)


def depth2batch(tensor: torch.Tensor, batch=0):
    return (
        tensor
        if tensor.ndim < 5
        else torch.swapaxes(tensor[batch], 0, 1).reshape(-1, *tensor.shape[-2:])[
            :, None
        ]
    )




def make_comparison(x: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    x = dicom_batch_to_video_batch(x)
    pred = dicom_batch_to_video_batch(pred)
    comp = torch.cat([depth2batch(img)[:32] for img in (x, pred, (x-pred).abs())])
    return comp


def log_comparison(x: torch.Tensor, pred: torch.Tensor, experiment, step: int):
    comp: torch.Tensor = make_comparison(x, pred)
    grid = make_grid(comp, nrow=32, normalize=True)
    experiment.add_image(
        tag="Comparison of the input (top), reconstruction/generation (middle) and difference between them (bottom)", 
        img_tensor=grid, # flipt from horizontal to vertical view,
        global_step=step,
    )

from torchmetrics.image.fid import FrechetInceptionDistance
# generate two slightly overlapping image intensity distributions

def calculate_fid(x: torch.Tensor, pred: torch.Tensor, normalize: bool=True) -> torch.Tensor:
    def ensure_three_channels(tensor):
        if tensor.size(1) == 1:  # Check if the tensor is grayscale
            tensor = tensor.repeat(1, 3, 1, 1)  # Repeat the single channel 3 times
        return tensor

    def __calculate_fid(pred, x, normalize=True):
        x = ensure_three_channels(x)
        pred = ensure_three_channels(pred)
        fid = FrechetInceptionDistance(normalize=normalize).to(x.device)
        fid.update(x, real=True)
        fid.update(pred, real=False)
        return fid.compute()


    # Ensure both x and pred have 3 channels
    
    loss_fn = __calculate_fid

    if pred.ndim == 5: # 3D Image: Just use 2D model and compute average over slices 
        depth = pred.shape[2] 
        losses = [loss_fn(pred[:,:,d], x[:,:,d], normalize=normalize) for d in range(depth)]
        losses = torch.stack(losses)
        return torch.mean(losses)
    else:
        return loss_fn(pred, x, normalize=normalize)


def calculate_lpips(x: torch.Tensor, pred: torch.Tensor, normalize: bool=False, linear_calibration: bool=False) -> torch.Tensor:
    loss_fn = lpips.LPIPS(net='vgg', lpips=linear_calibration).to(x.device) # Note: only 'vgg' valid as loss  
    # normalize # If true, normalize [0, 1] to [-1, 1]
        
    # No need to do that because ScalingLayer was introduced in version 0.1 which does this indirectly  
    # if pred.shape[1] == 1: # convert 1-channel gray images to 3-channel RGB
    #     pred = torch.concat([pred, pred, pred], dim=1)
    # if target.shape[1] == 1: # convert 1-channel gray images to 3-channel RGB 
    #     target = torch.concat([target, target, target], dim=1)

    if pred.ndim == 5: # 3D Image: Just use 2D model and compute average over slices 
        depth = pred.shape[2] 
        losses = [loss_fn(pred[:,:,d], x[:,:,d], normalize=normalize) for d in range(depth)]
        print(len(losses))
        losses = torch.stack(losses, dim=2)
        return torch.mean(losses, dim=2, keepdim=True)
    else:
        return loss_fn(pred, x, normalize=normalize)


def calculate_common_metrics(x: torch.Tensor, pred: torch.Tensor) -> dict:
    logging_dict = {}
    with torch.no_grad():
        logging_dict["L2"] = torch.nn.functional.mse_loss(pred, x)
        logging_dict["L1"] = torch.nn.functional.l1_loss(pred, x)
        logging_dict["ssim"] = ssim(
            (pred + 1) / 2, (x.type(pred.dtype) + 1) / 2, data_range=1
        )
        logging_dict["lpips"] = calculate_lpips(x, pred)
        # logging_dict["fid"] = calculate_fid(x, pred)
    
    return logging_dict


def log_visualizations(
    lightning_module: pl.LightningModule,
    x: torch.Tensor,
    x_out: torch.Tensor,
):
    if not lightning_module.trainer.is_global_zero:
        return
    log_comparison(
        x, x_out, lightning_module.logger.experiment, lightning_module.global_step
    )
    plot_2d_or_3d_image(
        x_out,
        step=lightning_module.global_step,
        writer=lightning_module.logger.experiment,
        frame_dim=-1,
        tag="Generated_scan",
    )
    plot_2d_or_3d_image(
        (x-x_out).abs(),
        step=lightning_module.global_step,
        writer=lightning_module.logger.experiment,
        frame_dim=-1,
        tag="Diff between input and generated",
    )

def log_common_metrics(
    lightning_module: pl.LightningModule,
    x: torch.Tensor,
    pred: torch.Tensor,
    stage: Literal["train", "test", "val"],
):
    # ----------------- Log Scalars ----------------------
    for metric_name, metric_val in calculate_common_metrics(x, pred).items():
        lightning_module.log(
            f"{stage}/{metric_name}",
            metric_val,
            batch_size=x.shape[0],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
