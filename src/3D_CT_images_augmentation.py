### CODE OF ANOTHER STUDENT. USED ONLY FOR RESEARCH PURPOSE.
# %% noq E402
import glob
import logging
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.apps import get_logger
from monai.data import CacheDataset
from monai.engines import GanTrainer
from monai.engines.utils import GanKeys, default_make_latent
from monai.handlers import CheckpointSaver, MetricLogger, StatsHandler
from monai.networks import normal_init
from monai.networks.nets import Discriminator, Generator
from monai.transforms import (
    AdjustContrast,
    CenterSpatialCrop,
    Compose,
    EnsureChannelFirst,
    EnsureType,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    Resize,
    ScaleIntensity,
)
from monai.utils import first, set_determinism
from monai.visualize import matshow3d
from torchsummary import summary

# %%
# load data to cache dataset
data_dir = "./data/input/"
directory = os.path.join(data_dir, "ct_images")
images_pattern = os.path.join(directory, "*.nii.gz")
images = sorted(glob.glob(images_pattern))

# %%
every_n_matshow3d = 8
cube_len = 256
depth = 128
batch_size = 1
latent_size = 128
disc_train_steps = 1
max_epochs = 2000
images_number = len(images)

train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        CenterSpatialCrop((390, 390, 0)),
        Resize((cube_len, cube_len, depth)),
        ScaleIntensity(),
        AdjustContrast(1.5),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        EnsureType(),
    ]
)

# %%
real_dataset = CacheDataset(images, train_transforms)
real_loader = torch.utils.data.DataLoader(
    real_dataset,
    batch_size=batch_size,
    num_workers=10,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
)

# %%
image_sample = first(real_loader)
print(image_sample.shape)

fig = plt.figure(figsize=(15, 15))
matshow3d(
    volume=image_sample,
    fig=fig,
    title="Sample image",
    every_n=every_n_matshow3d,
    frame_dim=-1,
    cmap="gray",
)

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(1))

# %%
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
get_logger("train_log")
set_determinism(0)
device = torch.device("cuda:0")

torch.cuda.empty_cache()
print(torch.cuda.memory_stats())
print(torch.cuda.memory_summary())

run_name = f"run_{max_epochs}_epochs_{cube_len}_size_{images_number}_images_{latent_size}_latent_{batch_size}_batch" + datetime.now().strftime(
    "%Y.%m.%d-%H:%M:%S"
)
print(run_name)

disc_net = Discriminator(
    in_shape=(1, cube_len, cube_len, depth),
    channels=(8, 16, 32, 64, 1),
    strides=(2, 2, 2, 2, 1),
    kernel_size=5,
    norm="BATCH",
).to(device)


gen_net = Generator(
    # latent_shape - tuple of integers stating the dimension of the input latent vector (minus batch dimension)
    # Input is first passed through a torch.nn.Linear layer to convert the input vector
    # to an image tensor with dimensions start_shape.
    latent_shape=latent_size,
    # start_shape - tuple of integers stating the dimension of the tensor to pass to convolution subnetwork
    start_shape=(256, 8, 8, 4),
    # channels - tuple of integers stating the output channels of each convolutional layer
    channels=[128, 64, 32, 16, 8, 1],
    # strides - tuple of integers stating the stride (upscale factor) of each convolutional layer
    strides=[2, 2, 2, 2, 2, 1],
    # kernel size - integer or tuple of integers stating size of convolutional kernels
    kernel_size=3,
    norm="BATCH",
)

gen_net.conv.add_module("activation", torch.nn.Sigmoid())
gen_net = gen_net.to(device)

# initialize both networks
disc_net.apply(normal_init)
gen_net.apply(normal_init)

# define optimizors
learning_rate = 2e-4
betas = (0.5, 0.999)
disc_opt = torch.optim.Adam(disc_net.parameters(), learning_rate, betas=betas)
gen_opt = torch.optim.Adam(gen_net.parameters(), learning_rate, betas=betas)

# define loss functions
disc_loss_criterion = torch.nn.BCELoss()
gen_loss_criterion = torch.nn.BCELoss()
real_label = 1
fake_label = 0

metric_logger = MetricLogger(
    loss_transform=lambda x: {
        GanKeys.GLOSS: x[GanKeys.GLOSS],
        GanKeys.DLOSS: x[GanKeys.DLOSS],
    },
    metric_transform=lambda x: x,
)

# create directory to store checkpoints
os.mkdir(f"/data2/etude/micorl/logs/{run_name}")
os.mkdir(f"/data2/etude/micorl/logs/{run_name}/checkpoints")

handlers = [
    StatsHandler(
        name="batch_training_loss",
        output_transform=lambda x: {
            GanKeys.GLOSS: x[GanKeys.GLOSS],
            GanKeys.DLOSS: x[GanKeys.DLOSS],
        },
    ),
    CheckpointSaver(
        save_dir=f"/data2/etude/micorl/logs/{run_name}/checkpoints/",
        save_dict={
            "g_net": gen_net,
            "d_net": disc_net,
            "disc_opt": disc_opt,
            "gen_opt": gen_opt,
        },
        save_interval=5,
        save_final=True,
        epoch_level=True,
    ),
    metric_logger,
]


print(summary(disc_net, (1, cube_len, cube_len, depth), 1))
print(summary(gen_net, (latent_size,), 1))


def prepare_batch(batchdata, device=None, non_blocking=False):
    return batchdata.to(device=device, non_blocking=non_blocking)


def discriminator_loss(gen_images, real_images):
    """
    The discriminator loss if calculated by comparing its
    prediction for real and generated images.
    """
    real = real_images.new_full((real_images.shape[0], 1), real_label)
    gen = gen_images.new_full((gen_images.shape[0], 1), fake_label)

    realloss = disc_loss_criterion(disc_net(real_images), real)
    genloss = disc_loss_criterion(disc_net(gen_images.detach()), gen)

    return (genloss + realloss) / 2


def generator_loss(gen_images):
    """
    The generator loss is calculated by determining how well
    the discriminator was fooled by the generated images.
    """
    output = disc_net(gen_images)
    cats = output.new_full(output.shape, real_label)
    return gen_loss_criterion(output, cats)


trainer = GanTrainer(
    device,
    max_epochs,
    real_loader,
    gen_net,
    gen_opt,
    generator_loss,
    disc_net,
    disc_opt,
    discriminator_loss,
    d_prepare_batch=prepare_batch,
    d_train_steps=disc_train_steps,
    g_update_latents=True,
    latent_shape=latent_size,
    key_train_metric=None,
    train_handlers=handlers,
)
trainer.run()

os.mkdir(f"/data2/etude/micorl/logs/{run_name}/metrics")

g_loss = [loss[1][GanKeys.GLOSS] for loss in metric_logger.loss]
d_loss = [loss[1][GanKeys.DLOSS] for loss in metric_logger.loss]

with open(f"/data2/etude/micorl/logs/{run_name}/metrics", "w") as f:
    for loss in metric_logger.loss:
        f.write(f"{loss[1][GanKeys.GLOSS]} {loss[1][GanKeys.DLOSS]}\n")

plt.figure(figsize=(12, 5))
plt.semilogy(g_loss, label="Generator Loss")
plt.semilogy(d_loss, label="Discriminator Loss")
plt.grid(True, "both", "both")
plt.legend()

fig = plt.figure(figsize=(15, 15))
test_latents = default_make_latent(1, latent_size).to(device)
fake_sample = gen_net(test_latents)

matshow3d(
    volume=fake_sample[0, 0],
    fig=fig,
    title="Generated sample",
    every_n=every_n_matshow3d,
    frame_dim=-1,
    cmap="gray",
)
