import matplotlib.pyplot as plt
from monai.visualize import matshow3d
from pathlib import Path
import nibabel as nib
import numpy as np
import torch


def plot_dicom(volume, title="Scan"):
    fig = plt.figure(figsize=(1, 32))
    matshow3d(volume=volume, fig=fig, title=title, every_n=1, frame_dim=-1, cmap="gray")

    return fig


def checkpoints_dir_path(model_name: str) -> Path:
    directory = Path(f"./models/{model_name}/checkpoints/")
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def tensor_to_nii(tensor: torch.Tensor, file_path):
    # Ensure the tensor is on the CPU and convert to NumPy array
    np_array = tensor.cpu().numpy()

    # Create a NIfTI image
    nifti_img = nib.Nifti1Image(np_array, affine=np.eye(4))

    # Save the NIfTI image to the specified file path
    nib.save(nifti_img, file_path)
