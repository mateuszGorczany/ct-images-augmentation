from enum import Enum, auto
from typing import Protocol

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pydantic.dataclasses import dataclass


class Stage(Enum):
    TRAIN = auto()
    TEST = auto()
    VAL = auto()


@dataclass
class ModelParameters:
    learning_rate: float = 0.1
    epochs: int = 10


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


class ExperimentTracker(Protocol):
    def log_metric(self, metric_name: str, metric_value: float) -> None:
        ...

    # and so on


@hydra.main(version_base=None, config_path="./../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    model_params = ModelParameters(**cfg.model)
    model = VisionTransformer(model_params)
    print(model_params)
    print(model)


if __name__ == "__main__":
    main()


def function1():
    return "function1"
