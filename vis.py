import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import einops
from scipy.ndimage import map_coordinates
from pathlib import Path
from typing import Tuple
from common import identity_grid
import matplotlib.animation as animation
import typer

app = typer.Typer()

@app.command()
def plot_animation(moving_image: Path, fixed_image: Path, flow_field: Path, save: Path=Path("animation.mp4"), animation_steps:int= 10, animation_speed: int=10):
    fixed_np = nib.load(fixed_image).get_fdata()
    moving_np = nib.load(moving_image).get_fdata()
    flow_field = nib.load(flow_field).get_fdata()
    middle = fixed_np.shape[-2]//2

    flow_formatted = einops.rearrange(flow_field, 'h d w n -> n h d w')
    grid = identity_grid(fixed_np.shape)

    moved_nps = [map_coordinates(moving_np, grid+(flow_formatted* step/animation_steps)) for step in range(animation_steps)]

    fig, ax = plt.subplots()
    ax.axis('off')
    ims = [[ax.imshow(moved_np[:,middle,:].T, cmap='gray', animated=True)] for moved_np in moved_nps]
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(save)

app()
