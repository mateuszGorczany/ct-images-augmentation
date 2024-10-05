# %%
import numpy as np
import matplotlib.pyplot as plt
from monai.visualize import matshow3d
# %%
# create a figure of a 3D volume
volume = np.random.rand(10, 10, 10)
fig = plt.figure()
matshow3d(volume, fig=fig, title="3D Volume")
# 
plt.show()
# create a figure of a list of channel-first 3D volumes
volumes = [np.random.rand(1, 10, 10, 10), np.random.rand(1, 10, 10, 10)]
fig = plt.figure()
matshow3d(volumes, fig=fig, title="List of Volumes")
plt.show()
# %%
%load_ext tensorboard
# %%
import monai
from torch.utils.tensorboard import SummaryWriter
# %%
writer = SummaryWriter()
# %% 
monai.visualize.img2tensorboard.plot_2d_or_3d_image(
    volumes, 1, 
    writer, 
    index=0, 
    max_channels=1, 
    frame_dim=-3, 
    max_frames=24, 
    tag='output'
)

# %%
