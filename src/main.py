from monai.data.dataset import CacheDataset, PersistentDataset
import torch
from omegaconf import DictConfig, OmegaConf, ListConfig
from pydantic.dataclasses import dataclass
import wandb
from argparse import ArgumentParser
from pathlib import Path
import ast
import datetime

from lightning.pytorch.loggers import TensorBoardLogger
from wandb.sdk.lib.disabled import RunDisabled
from wandb.sdk.wandb_run import Run
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
from src.models.wgan import (
    GAN as WGAN,
    ModelParameters as WGANModelParameters,
)
from src.models.monai_diffuser import (
    Model as Diffuser,
    ModelParameters as MonaiDiffuserModelParameters,
)
from src.models.medical_diffusion.vq_gan_3d.model.vqgan import VQGAN, ModelParameters
from src.models.meta_diffuser import (
    Model as MetaDiffuser,
    ModelParameters as MetaDiffuserModelParameters,
)
from dacite import from_dict
from src.dataset import load_dataset, DatasetConfig
from torch.utils.data import DataLoader
import lightning as L

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from src.utils import checkpoints_dir_path
from typing import Literal

Mode = Literal["train", "retrain", "predict"]

MODELS_AND_CONFIGS = {
    "mini_gan": (MiniGAN, MiniGanModelParameters),
    "simple_gan": (SimpleGAN, SimpleGanModelParameters),
    "monai_autoencoder": (MonaiAutoencoder, MonaiAutoencoderModelParameters),
    "monai_diffuser": (Diffuser, MonaiDiffuserModelParameters),
    "meta_vqgan": (VQGAN, ModelParameters),
    "meta_diffuser": (MetaDiffuser, MetaDiffuserModelParameters),
    "wgan": (WGAN, WGANModelParameters),
}

Profiler = Literal["simple", "advanced", "pytorch"]


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    seed: int = 42
    device: str = "cuda"
    train_ratio: float = 0.9
    num_nodes: int = 1
    devices: int = -1  # -1 for all gpus, 0 for single gpu
    accelerator: str = "gpu"
    wandb_project: str = "ct-images"
    metric_to_monitor: str = "val/loss"
    monitored_metric_mode: str = "min"  # min or max
    float_precision: int = 32
    profiler: str | None = None
    training_strategy: str = "ddp_find_unused_parameters_true"


torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class WandbModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        checkpoint_path = self.best_model_path
        if not checkpoint_path or not Path(checkpoint_path).is_file():
            return
        print(f"Checkpoint saved at: {checkpoint_path}")
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(checkpoint_path, name="best_model.ckpt")
        wandb.log_artifact(artifact)
        wandb.log({"checkpoint_path": checkpoint_path})
        print(f"Checkpoint saved to wandb: {checkpoint_path}")


def generate_run_name(model_name: str) -> str:
    return f"{model_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"


class Runner:
    def __init__(
        self,
    ) -> None:
        pass

    def setup_data_loaders(
        self, training_config: TrainingConfig, dataset_config: DatasetConfig
    ):
        dataset: CacheDataset | PersistentDataset = load_dataset(dataset_config)
        train_len: int = int(training_config.train_ratio * len(dataset))
        train_loader = DataLoader(
            dataset[:train_len],
            batch_size=training_config.batch_size,
            num_workers=8,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
        )
        scan = next(iter(train_loader))
        print("Tensor shape: ", scan.shape)

        val_loader = DataLoader(
            dataset[train_len:],
            batch_size=training_config.batch_size,
            num_workers=8,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )
        # self.model = None
        # self.model_config = None
        # default_model_checkpoint = checkpoints_dir_path(cfg.name) / f"{cfg.name}.ckpt"
        # self.model_checkpoint: Path = default_model_checkpoint
        # if Path(model_checkpoint).is_file():
        #     self.model_checkpoint: Path = Path(model_checkpoint)
        # elif checkpoints_dir_path(cfg.name).joinpath(model_checkpoint).is_file():
        #     self.model_checkpoint: Path = checkpoints_dir_path(cfg.name).joinpath(
        #         model_checkpoint
        #     )

        return train_loader, val_loader

    def setup_trainer(
        self, model_name: str, training_config: TrainingConfig
    ) -> Trainer:
        return Trainer(
            logger=TensorBoardLogger("./runs", name=training_config.wandb_project),  # type: ignore
            default_root_dir=checkpoints_dir_path(model_name),
            accelerator=training_config.accelerator,
            # devices="auto",
            devices=training_config.devices,
            # devices=0,
            num_nodes=training_config.num_nodes,
            max_epochs=training_config.epochs,
            callbacks=[
                # EarlyStopping(
                #     monitor=training_config.metric_to_monitor,
                #     min_delta=0.0,  # minimum change in the monitored quantity to qualify as an improvement
                #     patience=30,  # number of checks with no improvement
                #     mode=training_config.monitored_metric_mode,
                # ),
                ModelCheckpoint(
                    monitor=training_config.metric_to_monitor,
                    dirpath=checkpoints_dir_path(model_name),
                    save_top_k=1,
                    save_last=True,
                    mode=training_config.monitored_metric_mode,
                    save_on_train_epoch_end=True,
                ),
                LearningRateMonitor("epoch"),
                # WandbModelCheckpoint(
                #     monitor=training_config.metric_to_monitor,
                #     mode=training_config.monitored_metric_mode,
                #     save_last=True,
                #     save_top_k=1,
                # ),
            ],
            # resume_from_checkpoint=self.model_checkpoint if self.mode == "retrain" else None,
            strategy=training_config.training_strategy,
            # log_every_n_steps=10,
            profiler=training_config.profiler,
            # gradient_clip_val=1.0,
            precision=training_config.float_precision,
        )

    def train(self, cfg: DictConfig | ListConfig) -> torch.nn.Module:
        training_config: TrainingConfig = from_dict(
            TrainingConfig, OmegaConf.to_container(cfg.training)  # type: ignore
        )  # type: ignore
        dataset_config: DatasetConfig = from_dict(
            DatasetConfig, OmegaConf.to_container(cfg.dataset)  # type: ignore
        )  # type: ignore

        trainer = self.setup_trainer(cfg.name, training_config)
        train_loader, val_loader = self.setup_data_loaders(
            training_config, dataset_config
        )
        model_class, model_config_class = MODELS_AND_CONFIGS[cfg.name]
        model_config = from_dict(
            model_config_class, OmegaConf.to_container(cfg.model)  # type: ignore
        )

        run: Run | RunDisabled = wandb.init(
            project=training_config.wandb_project,
            config=dict(cfg),
            name=generate_run_name(cfg.name),
            sync_tensorboard=True,
        )
        L.seed_everything(training_config.seed)  # To be reproducible
        model = model_class(model_config)
        trainer.fit(model, train_loader, val_loader)
        # Load best checkpoint after training
        # if trainer.checkpoint_callback.best_model_path is not None:
        #     model = model_class.load_from_checkpoint( # type: ignore
        #         trainer.checkpoint_callback.best_model_path
        #     )
        # model.config = model_config

        return model

    def __new_run(self, wandb_run_id: str):
        api = wandb.Api()
        old_run: Run = api.run(f"ct-images/{wandb_run_id}")
        run: Run | RunDisabled = wandb.init(
            project="ct-images",
            name=generate_run_name(old_run.config["name"]),
            sync_tensorboard=True,
            config=old_run.config,
        )
        return run

    def retrain(
        self,
        wandb_run_id: str,
        model_checkpoint: Path | str = "",
        retrain_learning_rate: float | None = None,
        retrain_epochs: int | None = None,
    ):
        is_resume_with_new_run = retrain_epochs and retrain_learning_rate
        run: Run | RunDisabled = (
            self.__new_run(wandb_run_id)
            if is_resume_with_new_run
            else wandb.init(
                project="ct-images",
                sync_tensorboard=True,
                id=wandb_run_id,
                resume="must",
            )
        )

        training_config: TrainingConfig = from_dict(
            TrainingConfig, ast.literal_eval(run.config["training"])
        )  # type: ignore
        dataset_config: DatasetConfig = from_dict(
            DatasetConfig, ast.literal_eval(run.config["dataset"])
        )  # type: ignore
        if not model_checkpoint:
            model_checkpoint: Path = Path(run.use_model("best_model.ckpt"))
        model_checkpoint = Path(model_checkpoint)
        model_name = run.config["name"]
        model_class, model_config_class = MODELS_AND_CONFIGS[model_name]
        model_config = from_dict(
            model_config_class, ast.literal_eval(run.config["model"])  # type: ignore
        )

        if not model_checkpoint.is_file():
            print("Not Found pretrained model at %s, ..." % model_checkpoint)
            import sys

            sys.exit(1)
        print("Found pretrained model at %s, loading..." % model_checkpoint)

        L.seed_everything(training_config.seed)  # To be reproducible

        if is_resume_with_new_run:
            training_config.epochs = retrain_epochs  # type: ignore
            model_config.learning_rate = retrain_learning_rate
            model = model_class.load_from_checkpoint(model_checkpoint, config=model_config)  # type: ignore
            trainer = self.setup_trainer(model_name, training_config)
            train_loader, val_loader = self.setup_data_loaders(
                training_config, dataset_config
            )
            trainer.fit(model, train_loader, val_loader)
            return model

        training_config.epochs = 15000
        trainer = self.setup_trainer(model_name, training_config)
        train_loader, val_loader = self.setup_data_loaders(
            training_config, dataset_config
        )
        model = model_class(model_config)
        trainer.fit(model, train_loader, val_loader, ckpt_path=model_checkpoint)
        return model

    # def test(self):
    #     return self.trainer.test(self.model, dataloaders=self.val_loader, verbose=False)

    # def predict(self, checkpoint: Path) -> None:
    #     model_name = checkpoint.parent.name
    #     model_class, _ = MODELS_AND_CONFIGS[model_name]
    #     model = model_class.load_from_checkpoint(checkpoint)
    #     return model.predict(
    #         dataloaders=self.val_loader,  # it is only for torch lightning to work, look at the models predict func
    #     )


def main(
    config: Path | str,
    mode: str,
    model_checkpoint: Path,
    wandb_run_id: str,
    retrain_learning_rate: float | None,
    retrain_epochs: int | None,
) -> None:
    runner = Runner()
    match mode:
        case "train":
            with open(config, "r", encoding="utf8") as fp:
                config_yaml = fp.read()
            cfg = OmegaConf.create(config_yaml)
            OmegaConf.resolve(cfg)
            print(OmegaConf.to_yaml(cfg))
            runner.train(cfg)
        case "retrain":
            runner.retrain(
                model_checkpoint=Path(model_checkpoint),
                wandb_run_id=wandb_run_id,
                retrain_learning_rate=retrain_learning_rate,
                retrain_epochs=retrain_epochs,
            )
        # case "predict":
        #     runner.predict()

    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="", type=str, required=False)
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "retrain", "predict"]
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="",
        required=False,
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default="",
        required=False,
    )
    parser.add_argument(
        "--retrain-learning-rate",
        type=str,
        default="",
        required=False,
    )
    parser.add_argument(
        "--retrain-epochs",
        type=str,
        default="",
        required=False,
    )

    args = parser.parse_args()
    main(
        config=args.config,
        mode=args.mode,
        model_checkpoint=args.model_checkpoint,
        wandb_run_id=args.wandb_run_id,
        retrain_learning_rate=float(args.retrain_learning_rate)
        if args.retrain_learning_rate
        else None,
        retrain_epochs=int(args.retrain_epochs) if args.retrain_epochs else None,
    )
