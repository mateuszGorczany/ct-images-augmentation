from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from monai.networks.blocks import UnetOutBlock
import pytorch_lightning as pl

from src.models.transformers_ct_reconstruction.vqgan.conv_blocks import DownBlock, UpBlock, BasicBlock, BasicResBlock, UnetResBlock, UnetBasicBlock
from src.models.transformers_ct_reconstruction.vqgan.gan_losses import hinge_d_loss
from src.models.transformers_ct_reconstruction.vqgan.perceivers import LPIPS
from src.models.transformers_ct_reconstruction.vqgan.model_base import BasicModel, VeryBasicModel
from src.models.transformers_ct_reconstruction.vqgan.model import DiagonalGaussianDistribution

from pytorch_msssim import SSIM, ssim

from pydantic.dataclasses import dataclass


@dataclass
class ModelParameters:
    in_channels=3
    out_channels=3
    spatial_dims=2
    emb_channels=4
    hid_chs=[64, 128,  256, 512]
    kernel_sizes=[3,  3,   3,    3]
    strides=[1,  2,   2,   2]
    norm_name=("GROUP", {'num_groups': 8, "affine": True})
    act_name=("Swish", {})
    dropout=None
    use_res_block=True
    deep_supervision=False
    learnable_interpolation=True
    use_attention='none'
    embedding_loss_weight=1e-6
    perceiver=LPIPS
    perceiver_kwargs={}
    perceptual_loss_weight=1.0

    learning_rate=1e-3
    lr_scheduler=None
    lr_scheduler_kwargs={}
    loss=torch.nn.L1Loss
    loss_kwargs={'reduction': 'none'}

    sample_every_n_steps=1000

class Encoder(nn.Module):
    def __init__(
        self,
        config: ModelParameters,
        ConvBlock: UnetBasicBlock | UnetResBlock
    ):
        super().__init__()
        self.config = config
        self.depth = len(self.config.strides)
        use_attention = use_attention if isinstance(self.config.use_attention, list) else [
            self.config.use_attention]*len(self.config.strides)
        self.inc = ConvBlock(
            self.config.spatial_dims,
            self.config.in_channels,
            self.config.hid_chs[0],
            kernel_size=self.config.kernel_sizes[0],
            stride=self.config.strides[0],
            act_name=self.config.act_name,
            norm_name=self.config.norm_name,
            emb_channels=None
        )
        # ----------- Encoder ----------------
        self.encoders = nn.ModuleList([
            DownBlock(
                spatial_dims=self.config.spatial_dims,
                in_channels=self.config.hid_chs[i-1],
                out_channels=self.config.hid_chs[i],
                kernel_size=self.config.kernel_sizes[i],
                stride=self.config.strides[i],
                downsample_kernel_size=self.config.kernel_sizes[i],
                norm_name=self.config.norm_name,
                act_name=self.config.act_name,
                dropout=self.config.dropout,
                use_res_block=self.config.use_res_block,
                learnable_interpolation=self.config.learnable_interpolation,
                use_attention=use_attention[i],
                emb_channels=None
            )
            for i in range(1, self.depth)
        ])

        # ----------- Out-Encoder ------------
        self.out_enc = nn.Sequential(
            BasicBlock(self.config.spatial_dims, self.config.hid_chs[-1], 2*self.config.emb_channels, 3),
            BasicBlock(self.config.spatial_dims, 2*self.config.emb_channels, 2*self.config.emb_channels, 1)
        )

        # ----------- Reparameterization --------------
        # self.quantizer = DiagonalGaussianDistribution()

    def encode2(self, x):
        h = self.inc(x)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)
        # z, _ = self.quantizer(z)
        return z
    
    def encode(self, x):
        # --------- Encoder --------------
        h = self.inc(x)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)
        return z

    def forward(self, x):
        return self.encode(x)

class Decoder(nn.Module):
    def __init__(
        self,
        config: ModelParameters,
        ConvBlock: UnetBasicBlock | UnetResBlock
    ):
        self.config = config
        self.inc_dec = ConvBlock(
            self.config.spatial_dims, self.config.emb_channels, self.config.hid_chs[-1], 3, act_name=self.config.act_name, norm_name=self.config.norm_name)
        
        self.depth = len(self.config.strides)
        use_attention = use_attention if isinstance(self.config.use_attention, list) else [
            self.config.use_attention]*len(self.config.strides)

        self.decoders = nn.ModuleList([
            UpBlock(
                spatial_dims=self.config.spatial_dims,
                in_channels=self.config.hid_chs[i+1],
                out_channels=self.config.hid_chs[i],
                kernel_size=self.config.kernel_sizes[i+1],
                stride=self.config.strides[i+1],
                upsample_kernel_size=self.config.strides[i+1],
                norm_name=self.config.norm_name,
                act_name=self.config.act_name,
                dropout=self.config.dropout,
                use_res_block=self.config.use_res_block,
                learnable_interpolation=self.config.learnable_interpolation,
                use_attention=use_attention[i],
                emb_channels=None,
                skip_channels=0
            )
            for i in range(self.depth-1)
        ])

        # --------------- Out-Convolution ----------------
        self.outc = BasicBlock(
            self.config.spatial_dims, self.config.hid_chs[0], self.config.out_channels, 1, zero_conv=True)
        if isinstance(deep_supervision, bool):
            deep_supervision = self.depth-1 if deep_supervision else 0
        self.outc_ver = nn.ModuleList([
            BasicBlock(self.config.spatial_dims, self.config.hid_chs[i],
                       self.config.out_channels, 1, zero_conv=True)
            for i in range(1, deep_supervision+1)
        ])

    def decode(self, z_q):
        out_hor = []
        h = self.inc_dec(z_q)
        for i in range(len(self.decoders)-1, -1, -1):
            out_hor.append(self.outc_ver[i](h)) if i < len(
                self.outc_ver) else None
            h = self.decoders[i](h)
        out = self.outc(h)

        return out, out_hor

# @dataclass
# class QuantizerModelParams:
# class Quantizer(nn.Module):
    # def __init__(
    #     self,
    #     config: QuantizerModelParameters
    # ):

class VQVAE(pl.LightningModule):
    def __init__(
        self,
        config: ModelParameters
    ):
        super().__init__()
        # self.sample_every_n_steps = sample_every_n_steps
        self.config: ModelParameters = config
        self.loss_fct = torch.nn.L1Loss(reduction="none")
        ConvBlock: UnetResBlock | UnetBasicBlock = UnetResBlock if config.use_res_block else UnetBasicBlock
        self.encoder = Encoder(config, ConvBlock)
        self.decoder = Decoder(config, ConvBlock)
        # self.ssim_fct = SSIM(data_range=1, size_average=False, channel=out_channels, spatial_dims=spatial_dims, nonnegative_ssim=True)
        # self.embedding_loss_weight = embedding_loss_weight
        self.perceiver = LPIPS().eval()
        self.perceptual_loss_weight = self.config.perceptual_loss_weight

        # -------- Loss-Reg---------
        # self.logvar = nn.Parameter(torch.zeros(size=()) )

        # ----------- In-Convolution ------------


        # ----------- In-Decoder ------------

        # ------------ Decoder ----------
        # self.logvar_ver = nn.ParameterList([
        #     nn.Parameter(torch.zeros(size=()) )
        #     for _ in range(1, deep_supervision+1)
        # ])


    def decode(self, z):
        h = self.inc_dec(z)
        for i in range(len(self.decoders), 0, -1):
            h = self.decoders[i-1](h)
        x = self.outc(h)
        return x

    def forward(self, x_in):
        # --------- Encoder --------------
        z = self.encoder.encode(x_in)
        # --------- Quantizer --------------
        z_q, emb_loss = self.quantizer(z)

        # -------- Decoder -----------
        out, out_hor = self.decoder.decode(z_q)

        return out, out_hor[::-1], emb_loss

    def perception_loss(self, pred, target, depth=0):
        if (self.perceiver is not None) and (depth < 2):
            self.perceiver.eval()
            return self.perceiver(pred, target)*self.perceptual_loss_weight
        else:
            return 0

    def ssim_loss(self, pred, target):
        return 1-ssim(((pred+1)/2).clamp(0, 1), (target.type(pred.dtype)+1)/2, data_range=1, size_average=False,
                      nonnegative_ssim=True).reshape(-1, *[1]*(pred.ndim-1))

    def rec_loss(self, pred, pred_vertical, target):
        interpolation_mode = 'nearest-exact'

        # Loss
        loss = 0
        rec_loss = self.loss_fct(
            pred, target)+self.perception_loss(pred, target)+self.ssim_loss(pred, target)
        # rec_loss = rec_loss/ torch.exp(self.logvar) + self.logvar # Note this is include in Stable-Diffusion but logvar is not used in optimizer
        loss += torch.sum(rec_loss)/pred.shape[0]

        for i, pred_i in enumerate(pred_vertical):
            target_i = F.interpolate(
                target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)
            rec_loss_i = self.loss_fct(
                pred_i, target_i)+self.perception_loss(pred_i, target_i)+self.ssim_loss(pred_i, target_i)
            # rec_loss_i = rec_loss_i/ torch.exp(self.logvar_ver[i]) + self.logvar_ver[i]
            loss += torch.sum(rec_loss_i)/pred.shape[0]

        return loss

    def _step(self, batch: dict, batch_idx: int, state: str, optimizer_idx: int):
        # ------------------------- Get Source/Target ---------------------------
        x = batch
        target = x

        # ------------------------- Run Model ---------------------------
        pred, pred_vertical, emb_loss = self(x)

        # ------------------------- Compute Loss ---------------------------
        loss = self.rec_loss(pred, pred_vertical, target)
        loss += emb_loss*self.embedding_loss_weight

        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            logging_dict = {'loss': loss, 'emb_loss': emb_loss}
            logging_dict['L2'] = torch.nn.functional.mse_loss(pred, target)
            logging_dict['L1'] = torch.nn.functional.l1_loss(pred, target)
            logging_dict['ssim'] = ssim(
                (pred+1)/2, (target.type(pred.dtype)+1)/2, data_range=1)
            # logging_dict['logvar'] = self.logvar

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in logging_dict.items():
            self.log(f"{state}/{metric_name}", metric_val,
                     batch_size=x.shape[0], on_step=True, on_epoch=True)

        # ----------------- Save Image ------------------------------
        if self.global_step != 0 and self.global_step % self.sample_every_n_steps == 0:
            log_step = self.global_step // self.sample_every_n_steps
            path_out = Path(self.logger.log_dir)/'images'
            path_out.mkdir(parents=True, exist_ok=True)
            # for 3D images use depth as batch :[D, C, H, W], never show more than 16+16 =32 images

            def depth2batch(image):
                return (image if image.ndim < 5 else torch.swapaxes(image[0], 0, 1))
            images = torch.cat([depth2batch(img)[:16] for img in (x, pred)])
            save_image(
                images, path_out/f'sample_{log_step}.png', nrow=x.shape[0], normalize=True)

        return loss

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx:int = 0 ):
        return self._step(batch, batch_idx, "train", self._step_train, optimizer_idx)

    def validation_step(self, batch: dict, batch_idx: int, optimizer_idx:int = 0):
        return self._step(batch, batch_idx, "val", self._step_val, optimizer_idx )

    def test_step(self, batch: dict, batch_idx: int, optimizer_idx:int = 0):
        return self._step(batch, batch_idx, "test", self._step_test, optimizer_idx)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.config.learning_rate)
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]
        return optimizer