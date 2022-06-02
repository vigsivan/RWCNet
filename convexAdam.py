"""
Adapted from code written by Mattias Paul
"""

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
import einops
from scipy.ndimage.interpolation import zoom as zoom
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import typer

from common import FileMapping, fmapping_func

app = typer.Typer()

# correlation layer: dense discretised displacements to compute SSD cost volume with box-filter
def correlate(
    mind_fix: torch.Tensor,
    mind_mov: torch.Tensor,
    disp_hw: int,
    grid_sp: int,
    image_shape: Tuple[int, int, int],
):
    H, W, D = image_shape
    torch.cuda.synchronize()
    C_mind = mind_fix.shape[1]
    with torch.no_grad():
        mind_unfold = F.unfold(
            F.pad(
                mind_mov, (disp_hw, disp_hw, disp_hw, disp_hw, disp_hw, disp_hw)
            ).squeeze(0),
            disp_hw * 2 + 1,
        )
        mind_unfold = mind_unfold.view(
            C_mind, -1, (disp_hw * 2 + 1) ** 2, W // grid_sp, D // grid_sp
        )

    ssd = torch.zeros(
        (disp_hw * 2 + 1) ** 3,
        H // grid_sp,
        W // grid_sp,
        D // grid_sp,
        dtype=mind_fix.dtype,
        device=mind_fix.device,
    )  # .cuda().half()
    ssd_argmin = torch.zeros(H // grid_sp, W // grid_sp, D // grid_sp).long()
    with torch.no_grad():
        for i in range(disp_hw * 2 + 1):
            mind_sum = (
                (mind_fix.permute(1, 2, 0, 3, 4) - mind_unfold[:, i : i + H // grid_sp])
                .abs()
                .sum(0, keepdim=True)
            )

            ssd[i :: (disp_hw * 2 + 1)] = F.avg_pool3d(
                mind_sum.transpose(2, 1), 3, stride=1, padding=1
            ).squeeze(1)
        ssd = (
            ssd.view(
                disp_hw * 2 + 1,
                disp_hw * 2 + 1,
                disp_hw * 2 + 1,
                H // grid_sp,
                W // grid_sp,
                D // grid_sp,
            )
            .transpose(1, 0)
            .reshape((disp_hw * 2 + 1) ** 3, H // grid_sp, W // grid_sp, D // grid_sp)
        )
        ssd_argmin = torch.argmin(ssd, 0)  #
        # ssd = F.softmax(-ssd*1000,0)
    torch.cuda.synchronize()

    return ssd, ssd_argmin


# solve two coupled convex optimisation problems for efficient global regularisation


def coupled_convex(
    ssd: torch.Tensor,
    ssd_argmin: torch.Tensor,
    disp_mesh_t: torch.Tensor,
    grid_sp: int,
    image_shape: Tuple[int, int, int],
):
    H, W, D = image_shape
    disp_soft = F.avg_pool3d(
        disp_mesh_t.view(3, -1)[:, ssd_argmin.view(-1)].reshape(
            1, 3, H // grid_sp, W // grid_sp, D // grid_sp
        ),
        3,
        padding=1,
        stride=1,
    )

    coeffs = torch.tensor([0.003, 0.01, 0.03, 0.1, 0.3, 1])
    for j in range(6):
        ssd_coupled_argmin = torch.zeros_like(ssd_argmin)
        with torch.no_grad():
            for i in range(H // grid_sp):

                coupled = ssd[:, i, :, :] + coeffs[j] * (
                    disp_mesh_t - disp_soft[:, :, i].view(3, 1, -1)
                ).pow(2).sum(0).view(-1, W // grid_sp, D // grid_sp)
                ssd_coupled_argmin[i] = torch.argmin(coupled, 0)
            # print(coupled.shape)

        disp_soft = F.avg_pool3d(
            disp_mesh_t.view(3, -1)[:, ssd_coupled_argmin.view(-1)].reshape(
                1, 3, H // grid_sp, W // grid_sp, D // grid_sp
            ),
            3,
            padding=1,
            stride=1,
        )

    return disp_soft


# enforce inverse consistency of forward and backward transform
def inverse_consistency(
    disp_field1s: torch.Tensor, disp_field2s: torch.Tensor, iter: int = 20
):
    # factor = 1
    _, _, H, W, D = disp_field1s.size()
    # make inverse consistent
    with torch.no_grad():
        disp_field1i = disp_field1s.clone()
        disp_field2i = disp_field2s.clone()

        identity = (
            F.affine_grid(torch.eye(3, 4).unsqueeze(0), [1, 1, H, W, D])
            .permute(0, 4, 1, 2, 3)
            .to(disp_field1s.device)
            .to(disp_field1s.dtype)
        )
        for _ in range(iter):
            disp_field1s = disp_field1i.clone()
            disp_field2s = disp_field2i.clone()

            disp_field1i = 0.5 * (
                disp_field1s
                - F.grid_sample(
                    disp_field2s, (identity + disp_field1s).permute(0, 2, 3, 4, 1)
                )
            )
            disp_field2i = 0.5 * (
                disp_field2s
                - F.grid_sample(
                    disp_field1s, (identity + disp_field2s).permute(0, 2, 3, 4, 1)
                )
            )

    return disp_field1i, disp_field2i


def adam_optimization(
    disp_lr: torch.Tensor,
    mind_fixed: torch.Tensor,
    mind_moving: torch.Tensor,
    lambda_weight: float,
    grid_sp: int,
    image_shape: Tuple[int, int, int],
    iterations: int,
) -> nn.Module:

    H, W, D = image_shape
    # create optimisable displacement grid
    net = nn.Sequential(
        nn.Conv3d(3, 1, (H // grid_sp, W // grid_sp, D // grid_sp), bias=False)
    )
    net[0].weight.data[:] = disp_lr / grid_sp
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1)
    grid0 = F.affine_grid(
        torch.eye(3, 4).unsqueeze(0).cuda(),
        [1, 1, H // grid_sp, W // grid_sp, D // grid_sp],
        align_corners=False,
    )

    for _ in range(iterations):
        optimizer.zero_grad()

        disp_sample = F.avg_pool3d(
            F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1,
        ).permute(0, 2, 3, 4, 1)
        reg_loss = (
            lambda_weight
            * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean()
            + lambda_weight
            * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean()
            + lambda_weight
            * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()
        )

        scale = (
            torch.tensor(
                [
                    (H // grid_sp - 1) / 2,
                    (W // grid_sp - 1) / 2,
                    (D // grid_sp - 1) / 2,
                ]
            )
            .cuda()
            .unsqueeze(0)
        )
        grid_disp = (
            grid0.view(-1, 3).cuda().float()
            + ((disp_sample.view(-1, 3)) / scale).flip(1).float()
        )

        patch_mov_sampled = F.grid_sample(
            mind_moving.float(),
            grid_disp.view(1, H // grid_sp, W // grid_sp, D // grid_sp, 3).cuda(),
            align_corners=False,
            mode="bilinear",
        )  # ,padding_mode='border')
        sampled_cost = (patch_mov_sampled - mind_fixed).pow(2).mean(1) * 12

        loss = sampled_cost.mean()
        (loss + reg_loss).backward()
        optimizer.step()

    return net


@app.command()
def main(
    moving_directory: Path,
    fixed_directory: Path,
    seg_directory: Path,
    save_directory: Path,
    displacement_directory: Path,
    mov2fixed: FileMapping = FileMapping.identity,
    grid_sp: int = 2,
    disp_hw: int = 3,
    lambda_weight: float = 1.25,
    iterations: int = 100,
):
    """
    Performs intra-subject registration
    """

    save_directory.mkdir(exist_ok=True)

    moving_files = list(moving_directory.iterdir())
    displacement_directory.mkdir(exist_ok=True)
    mov2fixed_func = fmapping_func(mov2fixed)
    H, W, D = nib.load(moving_files[0]).shape

    for movingf in tqdm(moving_files):
        fixedf = fixed_directory / mov2fixed_func(movingf.name)

        torch.cuda.synchronize()

        fixed_nib = nib.load(fixedf)
        moving_nib = nib.load(movingf)

        fixed = torch.from_numpy(fixed_nib.get_fdata()).float()
        moving = torch.from_numpy(moving_nib.get_fdata()).float()

        fixed_seg = torch.from_numpy(
            nib.load(seg_directory / fixedf.name).get_fdata()
        ).float()
        moving_seg = torch.from_numpy(
            nib.load(seg_directory / movingf.name).get_fdata()
        ).float()

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
                    F.one_hot(fixed_seg.cuda().view(1, H, W, D).long())
                    .float()
                    .permute(0, 4, 1, 2, 3)
                    .contiguous()
                    * weight.view(1, -1, 1, 1, 1).cuda()
                ).half()
            )
            mindssc_mov_ = (
                10
                * (
                    F.one_hot(moving_seg.cuda().view(1, H, W, D).long())
                    .float()
                    .permute(0, 4, 1, 2, 3)
                    .contiguous()
                    * weight.view(1, -1, 1, 1, 1).cuda()
                ).half()
            )
            mind_fix_ = F.avg_pool3d(mindssc_fix_, grid_sp, stride=grid_sp)
            mind_mov_ = F.avg_pool3d(mindssc_mov_, grid_sp, stride=grid_sp)
            ssd, ssd_argmin = correlate(
                mind_fix_, mind_mov_, disp_hw, grid_sp, (H, W, D)
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
            disp_soft = coupled_convex(ssd, ssd_argmin, disp_mesh_t, grid_sp, (H, W, D))

        del ssd, mind_fix_, mind_mov_

        torch.cuda.empty_cache()

        disp_lr = F.interpolate(
            disp_soft * grid_sp,
            size=(H // 2, W // 2, D // 2),
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
            image_shape=(H, W, D),
            grid_sp=grid_sp,
            iterations=iterations,
        )

        torch.cuda.synchronize()

        disp_sample = F.avg_pool3d(
            F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1
        ).permute(0, 2, 3, 4, 1)
        fitted_grid = disp_sample.permute(0, 4, 1, 2, 3).detach()
        disp_hr = F.interpolate(
            fitted_grid * grid_sp, size=(H, W, D), mode="trilinear", align_corners=False
        )

        moved = torch.nn.functional.grid_sample(
                einops.repeat(moving, 'd h w -> n c d h w', n=1, c=1).cuda(),
                disp_hr.permute((0,2,3,4,1))
        ).squeeze().detach().cpu().numpy()

        # TODO: compute dice coefficient
        moved_seg = torch.nn.functional.grid_sample(
                einops.repeat(moving, 'd h w -> n c d h w', n=1, c=1).cuda(),
                disp_hr.permute((0,2,3,4,1))
        ).squeeze().detach().cpu().numpy()

        moved_nib = nib.Nifti1Image(moved, affine=fixed_nib.affine)
        nib.save(moved_nib, save_directory/movingf.name)

        disp_field = F.interpolate(
            disp_hr, scale_factor=0.5, mode="trilinear", align_corners=False
        )
        x1 = disp_field[0, 0, :, :, :].cpu().float().data.numpy()
        y1 = disp_field[0, 1, :, :, :].cpu().float().data.numpy()
        z1 = disp_field[0, 2, :, :, :].cpu().float().data.numpy()

        disp_name = f"{movingf.name}2{fixedf.name}.npz"
        displacement = np.stack((x1, y1, z1), 0)
        np.savez_compressed(displacement_directory / disp_name, displacement)

        torch.cuda.synchronize()


app()
