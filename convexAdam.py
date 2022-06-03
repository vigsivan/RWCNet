"""
Heavily adapted from code written by Mattias Paul
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import einops
import nibabel as nib
import numpy as np
from scipy.ndimage.interpolation import zoom as zoom
import torch
import torch.nn.functional as F
from tqdm import tqdm
import typer

from common import ( 
        MINDSSC,
        adam_optimization,
        apply_displacement_field,
        correlate,
        coupled_convex,
        data_generator 
)
from metrics import compute_dice

app = typer.Typer()

@app.command()
def train_without_labels(
    data_json: Path,
    save_directory: Path,
    grid_sp: int = 2,
    disp_hw: int = 3,
    lambda_weight: float = 1.25,
    iterations: int = 100,
    warp_images: bool = False,
):
    """
    Performs registration without labels.

    Parameters
    ----------
    data_json: Path,
        JSON containing the data paths. See definition in common.py
    save_directory: Path,
        Path where the outputs of this functin will be saved.
    grid_sp: int = 2,
        Grid spacing. Defualt=2
    disp_hw: int = 3,
    lambda_weight: float = 1.25,
    iterations: int = 100,
        Number of iterations of ADAM optimization. Default=100
    warp_images: bool = False,
        If True, the moving images are warped and saved in the save_directory. Default=False.
    """

    save_directory.mkdir(exist_ok=True)
    data_shape: Optional[Tuple[int, int, int]] = None
    gen = tqdm(data_generator(data_json))

    for data in gen:

        torch.cuda.synchronize()

        fixed_nib = nib.load(data.fixed_image)
        moving_nib = nib.load(data.moving_image)

        fixed = torch.from_numpy(fixed_nib.get_fdata()).float()
        moving = torch.from_numpy(moving_nib.get_fdata()).float()

        data_shape = fixed_nib.shape
        assert data_shape is not None

        torch.cuda.synchronize()

        with torch.no_grad():
            mindssc_fix_ = MINDSSC(fixed.unsqueeze(0).unsqueeze(0).cuda(), 1, 2).half()
            mindssc_mov_ = MINDSSC(moving.unsqueeze(0).unsqueeze(0).cuda(), 1, 2).half()

            mind_fix_ = F.avg_pool3d(mindssc_fix_, grid_sp, stride=grid_sp)
            mind_mov_ = F.avg_pool3d(mindssc_mov_, grid_sp, stride=grid_sp)
            ssd, ssd_argmin = correlate(
                mind_fix_, mind_mov_, disp_hw, grid_sp, data_shape
            )
            disp_mesh_t = (
                F.affine_grid(
                    disp_hw * torch.eye(3, 4).cuda().half().unsqueeze(0),
                    [1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1],
                    align_corners=True,
                )
                .permute(0, 4, 1, 2, 3)
                .reshape(3, -1, 1)
            )
            disp_soft = coupled_convex(
                ssd, ssd_argmin, disp_mesh_t, grid_sp, data_shape
            )

        del ssd, mind_fix_, mind_mov_

        torch.cuda.empty_cache()

        disp_lr = F.interpolate(
            disp_soft * grid_sp,
            size=tuple(s // 2 for s in data_shape),
            mode="trilinear",
            align_corners=False,
        )

        # extract one-hot patches
        torch.cuda.synchronize()

        with torch.no_grad():
            mind_fix_ = F.avg_pool3d(mindssc_fix_, grid_sp, stride=grid_sp)
            mind_mov_ = F.avg_pool3d(mindssc_mov_, grid_sp, stride=grid_sp)
        del mindssc_fix_, mindssc_mov_

        net = adam_optimization(
            disp_lr=disp_lr,
            mind_fixed=mind_fix_,
            mind_moving=mind_mov_,
            lambda_weight=lambda_weight,
            image_shape=data_shape,
            grid_sp=grid_sp,
            iterations=iterations,
        )

        torch.cuda.synchronize()

        disp_sample = F.avg_pool3d(
            F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1
        ).permute(0, 2, 3, 4, 1)
        fitted_grid = disp_sample.permute(0, 4, 1, 2, 3).detach()
        disp_hr = F.interpolate(
            fitted_grid * grid_sp,
            size=data_shape,
            mode="trilinear",
            align_corners=False,
        )

        disp_np = disp_hr.detach().cpu().numpy()

        # NOTE: we are using scipy's interpolate func, which does not take a batch dimension
        disp_np = einops.rearrange(disp_np, "b c d h w -> (b c) d h w")


        disp_name = f"{data.moving_image.name}2{data.fixed_image.name}"

        if warp_images:
            moved_image = apply_displacement_field(disp_np, moving.numpy())
            moved_nib = nib.Nifti1Image(moved_image, affine=fixed_nib.affine)
            nib.save(moved_nib, save_directory / data.moving_image.name)

        disp_field = F.interpolate(
            disp_hr, scale_factor=0.5, mode="trilinear", align_corners=False
        )
        x1 = disp_field[0, 0, :, :, :].cpu().float().data.numpy()
        y1 = disp_field[0, 1, :, :, :].cpu().float().data.numpy()
        z1 = disp_field[0, 2, :, :, :].cpu().float().data.numpy()

        displacement = np.stack([x1, y1, z1], 0)
        np.savez_compressed(save_directory / f"{disp_name}.npz", displacement)

        torch.cuda.synchronize()


@app.command()
def train_with_labels(
    data_json: Path,
    save_directory: Path,
    grid_sp: int = 2,
    disp_hw: int = 3,
    lambda_weight: float = 1.25,
    iterations: int = 100,
    warp_images: bool = False,
):
    """
    Performs registration with labels.

    Parameters
    ----------
    data_json: Path,
        JSON containing the data paths. See definition in common.py
    save_directory: Path,
        Path where the outputs of this functin will be saved.
    grid_sp: int = 2,
        Grid spacing. Defualt=2
    disp_hw: int = 3,
    lambda_weight: float = 1.25,
    iterations: int = 100,
        Number of iterations of ADAM optimization. Default=100
    warp_images: bool = False,
        If True, the moving images are warped and saved in the save_directory. Default=False.
    """

    save_directory.mkdir(exist_ok=True)
    data_shape: Optional[Tuple[int, int, int]] = None
    gen = tqdm(data_generator(data_json))
    dice_measurements = {}

    for data in gen:

        torch.cuda.synchronize()

        fixed_nib = nib.load(data.fixed_image)
        moving_nib = nib.load(data.moving_image)

        fixed = torch.from_numpy(fixed_nib.get_fdata()).float()
        moving = torch.from_numpy(moving_nib.get_fdata()).float()

        data_shape = fixed_nib.shape
        assert data_shape is not None

        fixed_seg = torch.from_numpy(
            nib.load(data.fixed_segmentation).get_fdata()
        ).float()
        moving_seg = torch.from_numpy(
            nib.load(data.moving_segmentation).get_fdata()
        ).float()


        # WTF!
        weight = 1 / (
            torch.bincount(fixed.long().reshape(-1))
            + torch.bincount(moving.long().reshape(-1))
        ).float().pow(0.3)
        weight /= weight.mean()

        torch.cuda.synchronize()

        with torch.no_grad():
            mindssc_fix_ = (
                10
                * (
                    F.one_hot(fixed_seg.cuda().view(1, *data_shape).long())
                    .float()
                    .permute(0, 4, 1, 2, 3)
                    .contiguous()
                    * weight.view(1, -1, 1, 1, 1).cuda()
                ).half()
            )
            mindssc_mov_ = (
                10
                * (
                    F.one_hot(moving_seg.cuda().view(1, *data_shape).long())
                    .float()
                    .permute(0, 4, 1, 2, 3)
                    .contiguous()
                    * weight.view(1, -1, 1, 1, 1).cuda()
                ).half()
            )
            mind_fix_ = F.avg_pool3d(mindssc_fix_, grid_sp, stride=grid_sp)
            mind_mov_ = F.avg_pool3d(mindssc_mov_, grid_sp, stride=grid_sp)
            ssd, ssd_argmin = correlate(
                mind_fix_, mind_mov_, disp_hw, grid_sp, data_shape
            )
            disp_mesh_t = (
                F.affine_grid(
                    disp_hw * torch.eye(3, 4).cuda().half().unsqueeze(0),
                    [1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1],
                    align_corners=True,
                )
                .permute(0, 4, 1, 2, 3)
                .reshape(3, -1, 1)
            )
            disp_soft = coupled_convex(
                ssd, ssd_argmin, disp_mesh_t, grid_sp, data_shape
            )

        del ssd, mind_fix_, mind_mov_

        torch.cuda.empty_cache()

        disp_lr = F.interpolate(
            disp_soft * grid_sp,
            size=tuple(s // 2 for s in data_shape),
            mode="trilinear",
            align_corners=False,
        )

        # extract one-hot patches
        torch.cuda.synchronize()

        with torch.no_grad():
            mind_fix_ = F.avg_pool3d(mindssc_fix_, grid_sp, stride=grid_sp)
            mind_mov_ = F.avg_pool3d(mindssc_mov_, grid_sp, stride=grid_sp)
        del mindssc_fix_, mindssc_mov_

        net = adam_optimization(
            disp_lr=disp_lr,
            mind_fixed=mind_fix_,
            mind_moving=mind_mov_,
            lambda_weight=lambda_weight,
            image_shape=data_shape,
            grid_sp=grid_sp,
            iterations=iterations,
        )

        torch.cuda.synchronize()

        disp_sample = F.avg_pool3d(
            F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1
        ).permute(0, 2, 3, 4, 1)
        fitted_grid = disp_sample.permute(0, 4, 1, 2, 3).detach()
        disp_hr = F.interpolate(
            fitted_grid * grid_sp,
            size=data_shape,
            mode="trilinear",
            align_corners=False,
        )

        disp_np = disp_hr.detach().cpu().numpy()

        # NOTE: we are using scipy's interpolate func, which does not take a batch dimension
        disp_np = einops.rearrange(disp_np, "b c d h w -> (b c) d h w")
        moved_seg = apply_displacement_field(disp_np, moving_seg.numpy())

        # Fix any problems that may have arisen due to linear interpolation
        fixed_seg[fixed_seg > 0.5] = 1
        moving_seg[moving_seg > 0.5] = 1
        moved_seg[moved_seg > 0.5] = 1

        disp_name = f"{data.moving_image.name}2{data.fixed_image.name}"
        dice = compute_dice(
            fixed_seg.numpy(), moving_seg.numpy(), moved_seg, labels=[1]
        )
        dice_measurements[disp_name] = dice

        if warp_images:
            moved_image = apply_displacement_field(disp_np, moving.numpy())
            moved_nib = nib.Nifti1Image(moved_image, affine=fixed_nib.affine)
            nib.save(moved_nib, save_directory / data.moving_image.name)

        disp_field = F.interpolate(
            disp_hr, scale_factor=0.5, mode="trilinear", align_corners=False
        )
        x1 = disp_field[0, 0, :, :, :].cpu().float().data.numpy()
        y1 = disp_field[0, 1, :, :, :].cpu().float().data.numpy()
        z1 = disp_field[0, 2, :, :, :].cpu().float().data.numpy()

        displacement = np.stack([x1, y1, z1], 0)
        np.savez_compressed(save_directory / f"{disp_name}.npz", displacement)

        torch.cuda.synchronize()

    with open(save_directory / "dice_measurements.json", "w") as f:
        json.dump(dice_measurements, f)


app()
