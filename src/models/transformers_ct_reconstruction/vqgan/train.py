"""
Adapted from https://github.com/mueller-franzes/medfusion
"""

import argparse

import torch
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.models.transformers_ct_reconstruction.vqgan.model import VQVAE, VQGAN
from src.dataset import load_dataset
from monai.data.dataset import CacheDataset, PersistentDataset
from torch.utils.data import DataLoader
from pathlib import Path


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import wandb
from wandb.sdk.lib.disabled import RunDisabled
from wandb.sdk.wandb_run import Run
from src.main import generate_run_name, checkpoints_dir_path
from src.dataset import DatasetConfig
from lightning.pytorch.loggers import TensorBoardLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--best-vq-vae-ckpt", type=str, default=None, help="Path to the best checkpoint for the VQ-VAE model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --------------- Settings --------------------
    model_name = "german_vqgan" if args.best_vq_vae_ckpt else "german_vqvae"
    path_run_dir = checkpoints_dir_path(model_name)
    path_run_dir.mkdir(parents=True, exist_ok=True)

    dataset: CacheDataset | PersistentDataset = load_dataset(
        DatasetConfig(
            caching= "memory", # disk or memory caching
            path= "/ravana/d3d_work/micorl/data/ct_images_prostate_32fixed/",
            image_size= 128,  # image height and width
            num_slices= 32,  # image depth
            win_wid= 400,  # window width for converting to HO scale
            win_lev= 60,
        )
    )
    train_len: int = int(0.9* len(dataset))
    train_loader = DataLoader(
        dataset[:train_len],
        batch_size=5,
        num_workers=8,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    print(len(train_loader))
    scan = next(iter(train_loader))
    print("Tensor shape: ", scan.shape)

    val_loader = DataLoader(
        dataset[train_len:],
        batch_size=5,
        num_workers=8,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    run: Run | RunDisabled = wandb.init(
        project="ct-images",
        name=generate_run_name(model_name),
        sync_tensorboard=True,
    )
    if not args.best_vq_vae_ckpt:
        model = VQVAE(
        in_channels=1,
        out_channels=1,
        emb_channels=8,
        num_embeddings=16384,
        spatial_dims=3,
        hid_chs=[32, 64,  128, 256],
        kernel_sizes=[3,  3,   3, 3],
        strides=[1,  2,   2, 2],
        embedding_loss_weight=1,
        beta=1,
        loss=torch.nn.L1Loss,
        deep_supervision=0,
        # use_attention='none',
        use_attention=[False,False,True,True],
        norm_name=("GROUP", {'num_groups': 4, "affine": True}),
        sample_every_n_epochs=20,
        optimizer_kwargs={"lr": 5e-6}
        )
    else:
        model = VQGAN(
        in_channels=1,
        out_channels=1,
        emb_channels=256,
        num_embeddings=8192,
        spatial_dims=3,
        hid_chs=[32, 64,  128, 256],
        kernel_sizes=[3,  3,   3, 3],
        strides=[1,  2,   2, 2],
        embedding_loss_weight=1,
        beta=1,
        pixel_loss=torch.nn.L1Loss,
        deep_supervision=0,
        use_attention='none',
        norm_name=("GROUP", {'num_groups': 4, "affine": True}),
        sample_every_n_epochs=20,
        )

        model.vqvae.load_pretrained(Path(args.best_vq_vae_ckpt))

    ##############################################

    # -------------- Training Initialization ---------------
    to_monitor = "val/ssim_epoch"  # "train/L1"  # "val/loss"
    min_max = "max"
    save_and_sample_every = 50

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0,  # minimum change in the monitored quantity to qualify as an improvement
        patience=30,  # number of checks with no improvement
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),  # dirpath
        monitor=to_monitor,
        every_n_train_steps=save_and_sample_every,
        save_last=True,
        save_top_k=1,
        mode=min_max,
    )
    trainer = Trainer(
        logger=TensorBoardLogger("./runs", name="ct-images"),  # type: ignore
        accelerator='gpu',
        devices=-1,
        precision=32,
        # amp_backend='apex',
        # amp_level='O2',
        # gradient_clip_val=0.5,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        # callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=save_and_sample_every,
        # limit_train_batches=1000,
        # limit_val_batches=0,  # 0 = disable validation - Note: Early Stopping no longer available
        strategy="ddp",
        min_epochs=100,
        max_epochs=10001,
        num_sanity_val_steps=2,
    )

    # ---------------- Execute Training ----------------
    trainer.fit(model, train_loader, val_loader)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(
        trainer.logger.log_dir, checkpointing.best_model_path)
