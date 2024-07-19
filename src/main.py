import torch
from omegaconf import DictConfig, OmegaConf
from pydantic.dataclasses import dataclass
import wandb
from argparse import ArgumentParser
from src.models.vision_transformer import (
    VisionTransformer,
    ModelParameters as VisionTransformerModelParameters,
)
from src.models.simple_gan import (
    GAN as SimpleGAN,
    ModelParameters as SimpleGanModelParameters,
)
from dacite import from_dict
from src.dataset import load_dataset
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from src.utils import checkpoints_dir_path

MODELS_AND_CONFIGS = {
    "simple_gan": (SimpleGAN, SimpleGanModelParameters),
    "vision_transformer": (VisionTransformer, VisionTransformerModelParameters),
}


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: int
    epochs: int
    seed: int
    num_nodes: int
    device: str
    wandb_project = "ct-images"


torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class Runner:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.training_config: TrainingConfig = from_dict(
            TrainingConfig, OmegaConf.to_container(cfg.training)
        )

        self.run = wandb.init(
            project=self.training_config.wandb_project, name=cfg.name, config=dict(cfg)
        )

        self.dataset = load_dataset(cfg.dataset.name, cfg.dataset.kwargs)

        self.train_loader = DataLoader(
            self.dataset.train_split, batch_size=self.training_config.batch_size
        )
        self.val_loaders = {
            key: DataLoader(val_ds, batch_size=cfg.training.batch_size)
            for key, val_ds in self.dataset.validation_split.items()
        }

        self.train_metrics = self.dataset.train_metrics.to(device=cfg.training.device)
        self.val_metrics = self.dataset.validation_metrics.to(
            device=cfg.training.device
        )

        self.model = None
        self.model_config = None

    def train(self):
        self.trainer = L.Trainer(
            logger=L.pytorch.loggers.WandbLogger(self.training_config.wandb_project),
            default_root_dir=checkpoints_dir_path(cfg.name),
            accelerator="auto",
            devices=-1,
            num_nodes=self.training_config.num_nodes,
            max_epochs=self.training_config.epochs,
            callbacks=[
                ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                LearningRateMonitor("epoch"),
            ],
            fast_dev_run=True,
        )
        # self.trainer.logger._log_graph = (
        #     True  # If True, we plot the computation graph in tensorboard
        # )
        # self.trainer.logger._default_hp_metric = (
        #     None  # Optional logging argument that we don't need
        # )

        model_class, model_config_class = MODELS_AND_CONFIGS[cfg.name]

        # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_model_filename = checkpoints_dir_path(cfg.name) / f"{cfg.name}.ckpt"
        if pretrained_model_filename.is_file():
            print(
                "Found pretrained model at %s, loading..." % pretrained_model_filename
            )
            # Automatically loads the model with the saved hyperparameters
            self.model = model_class.load_from_checkpoint(pretrained_model_filename)
        else:
            L.seed_everything(self.training_config.seed)  # To be reproducible
            self.model_config = from_dict(
                model_config_class, OmegaConf.to_container(cfg.model)
            )
            self.model = model_class(self.model_config).to(
                device=self.training_config.device
            )

            self.model.reset_parameters()

            self.trainer.fit(self.model, self.train_loader, self.val_loader)
            # Load best checkpoint after training
            self.model = model_class.load_from_checkpoint(
                self.trainer.checkpoint_callback.best_model_path
            )

        return self.model

    def test(self):
        return self.trainer.test(
            self.model, dataloaders=self.test_loader, verbose=False
        )

    def validate(self):
        return self.trainer.test(self.model, dataloaders=self.val_loader, verbose=False)


def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # model = model.to(dtype=torch_dtype_map[cfg.training.weight_precision])

    # optim_groups = model._create_weight_decay_optim_groups()

    # optimizer = optim.AdamW(
    #     (
    #         {"weight_decay": cfg.training.weight_decay, "params": optim_groups[0]},
    #         {"weight_decay": 0.0, "params": optim_groups[1]},
    #     ),
    #     lr=cfg.training.lr,
    # )

    # lr_scheduler = LinearWarmupCosineAnnealing(
    #     optimizer,
    #     cfg.training.lr_warmup_steps,
    #     cfg.training.lr_decay_until_steps,
    #     cfg.training.lr,
    #     cfg.training.lr_decay_factor * cfg.training.lr,
    # )
    # Data

    runner = Runner(cfg)
    runner.train()
    # Test best model on validation and test set
    result = {
        "test": runner.test()[0]["test_acc"],
        "val": runner.validate()[0]["test_acc"],
    }

    print(result)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf8") as fp:
        config_yaml = fp.read()
    cfg = OmegaConf.create(config_yaml)
    OmegaConf.resolve(cfg)
    main(cfg)
