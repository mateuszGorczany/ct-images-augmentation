import matplotlib.pyplot as plt
from monai.visualize import matshow3d
from pathlib import Path


def plot_dicom(volume, title="Scan"):
    fig = plt.figure(figsize=(15,15))
    matshow3d(
        volume=volume,
        fig=fig,
        title=title,
        every_n=1,
        frame_dim=-1,
        cmap="gray"
    )
    
    return fig

def checkpoints_dir_path(model_name: str) -> Path:
    directory = Path(f"./models/{model_name}/checkpoints/")
    directory.mkdir(parents=True, exist_ok=True)

    return directory 

