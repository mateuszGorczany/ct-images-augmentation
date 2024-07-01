import torch
from pydantic.dataclasses import dataclass
from lightning import LightningModule
from torch import optim

@dataclass
class ModelParameters:
    pass

class VisionTransformer(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x


class LightningVisionTransformer(LightningModule):
    def __init__(self, params: ModelParameters):
        super().__init__()

        # self.network = torch.nn.Sequential()
        self.network = VisionTransformer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        # self.lr_scheduler = CosineWarmupScheduler(
        #     optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
        # )
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.lr_scheduler.step()  # Step per iteration

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        # compute loss
        loss = self.compute_loss(batch)

        opt.zero_grad()
        self.manual_backward(loss)

        # clip gradients
        self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        opt.step()
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError