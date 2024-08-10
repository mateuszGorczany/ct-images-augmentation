from monai.data.dataset import CacheDataset
import torch
from omegaconf import DictConfig, OmegaConf
from pydantic.dataclasses import dataclass
import wandb
from argparse import ArgumentParser

from lightning.pytorch.loggers import TensorBoardLogger
from wandb.sdk.lib.disabled import RunDisabled
from wandb.sdk.wandb_run import Run
from src.models.vision_transformer import (
    VisionTransformer,
    ModelParameters as VisionTransformerModelParameters,
)
from src.models.simple_gan import (
    GAN as SimpleGAN,
    ModelParameters as SimpleGanModelParameters,
)
from src.models.mini_gan import (
    GAN as MiniGAN,
    ModelParameters as MiniGanModelParameters,
)
from src.models.monai_autoencoder import (
    GAN as MonaiAutoencoder,
    ModelParameters as MonaiAutoencoderModelParameters,
)
from dacite import from_dict
from src.dataset import load_dataset, DatasetConfig
from torch.utils.data import DataLoader
import lightning as L

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from src.utils import checkpoints_dir_path
from typing import Literal

Mode = Literal["train", "retrain", "test"]

MODELS_AND_CONFIGS = {
    "mini_gan": (MiniGAN, MiniGanModelParameters),
    "simple_gan": (SimpleGAN, SimpleGanModelParameters),
    "vision_transformer": (VisionTransformer, VisionTransformerModelParameters),
    "monai_autoencoder": (MonaiAutoencoder, MonaiAutoencoderModelParameters),
}


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    seed: int
    num_nodes: int
    device: str
    wandb_project = "ct-images"
    train_ratio: float


torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class Runner:
    def __init__(self, cfg: DictConfig, mode: Mode = "train") -> None:
        self.cfg: DictConfig = cfg
        self.mode: Mode = mode
        self.training_config: TrainingConfig = from_dict(
            TrainingConfig, OmegaConf.to_container(cfg.training)
        )

        self.run: Run | RunDisabled = wandb.init(
            project=self.training_config.wandb_project,
            name=cfg.name,
            config=dict(cfg),
            sync_tensorboard=True,
        )

        dataset_config: DatasetConfig = from_dict(
            DatasetConfig, OmegaConf.to_container(cfg.dataset)
        )
        self.dataset: CacheDataset = load_dataset(dataset_config)

        train_len: int = int(self.training_config.train_ratio * len(self.dataset))

        self.train_loader = DataLoader(
            self.dataset[:train_len],
            batch_size=self.training_config.batch_size,
            num_workers=8,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )
        self.val_loader = DataLoader(
            self.dataset[train_len:],
            batch_size=self.training_config.batch_size,
            num_workers=8,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        self.model = None
        self.model_config = None

    def train(self):
        self.trainer = Trainer(
            # logger=WandbLogger(self.training_config.wandb_project),
            logger=TensorBoardLogger("./runs", name=self.training_config.wandb_project),
            default_root_dir=checkpoints_dir_path(cfg.name),
            accelerator=self.training_config.device,
            # devices="auto",
            devices=-1,
            num_nodes=self.training_config.num_nodes,
            max_epochs=self.training_config.epochs,
            callbacks=[
                ModelCheckpoint(
                    dirpath=checkpoints_dir_path(cfg.name),
                    filename=cfg.name,
                    mode="max",
                    save_on_train_epoch_end=True,
                ),
                LearningRateMonitor("epoch"),
            ],
            strategy="ddp_find_unused_parameters_true"
            # log_every_n_steps=10,
        )

        model_class, model_config_class = MODELS_AND_CONFIGS[cfg.name]

        # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_model_filename = checkpoints_dir_path(cfg.name) / f"{cfg.name}.ckpt"

        match self.mode:
            case "retrain":
                if not pretrained_model_filename.is_file():
                    print(
                        "Not Found pretrained model at %s, ..."
                        % pretrained_model_filename
                    )
                    import sys

                    sys.exit(1)
                else:
                    print(
                        "Found pretrained model at %s, loading..."
                        % pretrained_model_filename
                    )
                    # Automatically loads the model with the saved hyperparameters
                    self.model = model_class.load_from_checkpoint(
                        pretrained_model_filename
                    )

                    self.trainer.fit(self.model, self.train_loader, self.val_loader)

                    return self.model
            case "test":
                self.model = model_class.load_from_checkpoint(
                    self.trainer.checkpoint_callback.best_model_path
                )
                return self.model

            case "train":
                L.seed_everything(self.training_config.seed)  # To be reproducible
                self.model_config = from_dict(
                    model_config_class, OmegaConf.to_container(cfg.model)
                )
                self.model = model_class(self.model_config).to(
                    device=self.training_config.device
                )

                # self.model.reset_parameters()
                self.trainer.fit(self.model, self.train_loader, self.val_loader)
                # Load best checkpoint after training
                self.model = model_class.load_from_checkpoint(
                    self.trainer.checkpoint_callback.best_model_path
                )
                self.model.config = self.model_config

                return self.model

    def test(self):
        return self.trainer.test(self.model, dataloaders=self.val_loader, verbose=False)

    def validate(self):
        return self.trainer.test(self.model, dataloaders=self.val_loader, verbose=False)


def main(cfg: DictConfig, mode: str) -> None:
    print(OmegaConf.to_yaml(cfg))
    runner = Runner(cfg, mode=mode)
    runner.train()
    print(runner.test())
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "retrain", "test"]
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf8") as fp:
        config_yaml = fp.read()
    cfg = OmegaConf.create(config_yaml)
    OmegaConf.resolve(cfg)
    main(cfg, mode=args.mode)
