from enum import Enum, auto
from typing import Protocol

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pydantic.dataclasses import dataclass
import wandb


class Stage(Enum):
    TRAIN = auto()
    TEST = auto()
    VAL = auto()


@dataclass
class ModelParameters:
    learning_rate: float
    epochs: int


class VisionTransformer(torch.nn.Module):
    def __init__(self, params: ModelParameters):
        super().__init__()

        self.network = torch.nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Runner(Protocol):
    def train(self) -> None:
        ...

    def test(self) -> None:
        ...

    def validate(self) -> None:
        ...


class GanRunner(Runner):
    def __init__(
        self, model: torch.nn.Module, run: wandb.Run, run_config=ModelParameters
    ):
        self.model = model
        self.run = run
        self.run_config = run_config

    def train(self) -> None:
        for epoch in self.run_config.epochs:
            loss = ...
            self.run.log({"loss": loss})

    def test(self) -> None:
        # model()
        pass

    def validate(self) -> None:
        pass


class ExperimentTracker(Protocol):
    def log_metric(self, metric_name: str, metric_value: float) -> None:
        ...

    # and so on


@hydra.main(version_base=None, config_path="./../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    model_params = ModelParameters(**cfg.model)
    model_types: list[torch.nn.Module] = [VisionTransformer]

    for model_type in model_types:
        run = wandb.init(project="my_first_project", config=dict(model_params))
        model = model_type(model_params)
        runner = GanRunner(model, run, model_params)
        runner.train()
        runner.validate()
        runner.test()


if __name__ == "__main__":
    main()


def function1():
    return "function1"
