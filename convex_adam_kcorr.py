from collections import defaultdict
import json
from pathlib import Path

import einops
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import typer

from common import (
    DisplacementFormat,
    MINDSEG,
    MINDSSC,
    adam_optimization,
    apply_displacement_field,
    correlate,
    correlate_K,
    coupled_convex2,
    data_generator,
    get_labels,
)
from metrics import compute_dice, compute_total_registration_error
from differentiable_metrics import TotalRegistrationLoss

app = typer.Typer()

add_bc_dim = lambda x: einops.repeat(x, "d h w -> b c d h w", b=1, c=1).float()
get_spacing = lambda x: np.sqrt(np.sum(x.affine[:3, :3] * x.affine[:3, :3], axis=0))

@app.command()
def main(
    data_json: Path,
    save_directory: Path,
    grid_sp: int = 2,
    disp_hw: int = 3,
    use_labels: bool = True,
    split: str="train",
    lambda_weight: float = 1.25,
    iterations: int = 100,
    skip_normalize: bool = False,
    use_l2r_naming: bool=True,
    disp_format: DisplacementFormat=DisplacementFormat.Nifti,
    warp_images: bool = True,
    warp_segmentations: bool = True,
):
    """
    Performs registration with labels.

    By default,this function saves displacement files compatible with
    scipy's interpolation function at save_directory/disps as well as
    dice scores at save_directory/measurements.json.

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
        Regularization weight.
    iterations: int = 100,
        Number of iterations of ADAM optimization. Default=100
    compute_mind_from_seg: bool
        Computes features directly from the segmentation. Works well, but kinda iffy to read.
        Dafault True.
    skip_normalize: bool
        Skip image normalization. Normal intensity range is not required, 
        but functionality at the very least assumes a positive intensity range.
        Defualt: False.
    warp_images: bool = False,
        If True, the moving images are warped and saved in save_directory/images.
        Default=False.
    warp_segmentations: bool = False,
        If True, the moving segmentations are warped and saved in save_directory/segmentations.
        Default=False.
    """

    save_directory.mkdir(exist_ok=True)
    (save_directory / "disps").mkdir(exist_ok=True)

    if warp_images:
        warp_images_dir = save_directory / "images"
        warp_images_dir.mkdir(exist_ok=True)

    if warp_segmentations:
        warp_images_dir = save_directory / "segmentations"
        warp_images_dir.mkdir(exist_ok=True)

    gen = tqdm(data_generator(data_json, split=split))
    measurements = defaultdict(dict)

    for data in gen:

        torch.cuda.synchronize()

        fixed_nib = nib.load(data.fixed_image)
        moving_nib = nib.load(data.moving_image)

        fixed = add_bc_dim(torch.from_numpy(fixed_nib.get_fdata())).cuda().squeeze()
        moving = add_bc_dim(torch.from_numpy(moving_nib.get_fdata())).cuda().squeeze()


        if not skip_normalize:
            fixed = (fixed - fixed.min())/(fixed.max() - fixed.min())
            moving = (moving - moving.min())/(moving.max() - moving.min())

        data_shape = fixed_nib.shape


        torch.cuda.synchronize()

        with torch.no_grad():
            if use_labels:
                fixed_seg = add_bc_dim(
                    torch.from_numpy(
                        np.round(nib.load(data.fixed_segmentation).get_fdata())
                    )
                ).cuda().squeeze()
                moving_seg = add_bc_dim(
                    torch.from_numpy(
                        np.round(nib.load(data.moving_segmentation).get_fdata())
                    )
                ).cuda().squeeze()

                label_list = get_labels(fixed_seg, moving_seg)

                maxlabels = max(torch.unique(fixed_seg.long()).shape[0], torch.unique(moving_seg.long()).shape[0])
                weight = 1 / (
                    torch.bincount(fixed_seg.long().reshape(-1), minlength=maxlabels)
                    + torch.bincount(moving_seg.long().reshape(-1), minlength=maxlabels)
                ).float().pow(0.3)
                weight[torch.isinf(weight)]=0.
                weight /= weight.mean()
                mindssc_fix_ = MINDSEG(fixed_seg, data_shape, weight)
                mindssc_mov_ = MINDSEG(moving_seg, data_shape, weight)

            else:
                mindssc_fix_ = MINDSSC(
                    fixed.unsqueeze(0).unsqueeze(0).cuda(), 1, 2
                ).half()
                mindssc_mov_ = MINDSSC(
                    moving.unsqueeze(0).unsqueeze(0).cuda(), 1, 2
                ).half()

            mind_fix_ = F.avg_pool3d(mindssc_fix_, grid_sp, stride=grid_sp)
            mind_mov_ = F.avg_pool3d(mindssc_mov_, grid_sp, stride=grid_sp)
            ssd, ssd_argmin = correlate(
                mind_fix_, mind_mov_, disp_hw, grid_sp, data_shape
            )
            sims, disps = correlate_K(mind_fix_, mind_mov_, K=10)
            disp_mesh_t = (
                F.affine_grid(
                    disp_hw * torch.eye(3, 4).cuda().half().unsqueeze(0),
                    [1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1],
                    align_corners=True,
                )
                .permute(0, 4, 1, 2, 3)
                .reshape(3, -1, 1)
            )
            disp_soft = coupled_convex2(
                sims, disps, grid_sp, data_shape
            )

        del ssd, mind_fix_, mind_mov_

        torch.cuda.empty_cache()

        disp_lr = F.interpolate(
            disp_soft * grid_sp,
            size=tuple(s // grid_sp for s in data_shape),
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
            disp=disp_lr,
            mind_fixed=mind_fix_,
            mind_moving=mind_mov_,
            lambda_weight=lambda_weight,
            image_shape=tuple(s//grid_sp for s in data_shape),
            iterations=iterations,
            norm=grid_sp
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

        trl = TotalRegistrationLoss()

        disp_np = disp_hr.detach().cpu().numpy()

        # NOTE: we are using scipy's interpolate func, which does not take a batch dimension
        disp_np = einops.rearrange(disp_np, "b c d h w -> (b c) d h w")
        l2r_disp = einops.rearrange(disp_np, 't h w d -> h w d t')

        if use_l2r_naming:
            disp_name = f"disp_{data.fixed_image.name[-16:-12]}_{data.moving_image.name[-16:-12]}"
        else:
            disp_name = f"disp_{data.fixed_image.name.split('.')[0]}_{data.moving_image.name.split('.')[0]}"

        if use_labels:
            moving_seg = moving_seg.detach().cpu() #type: ignore
            fixed_seg = fixed_seg.detach().cpu() #type: ignore
            moved_seg = apply_displacement_field(disp_np, moving_seg.numpy(), order=0) 

            dice = compute_dice(
                fixed_seg.numpy(), moving_seg.numpy(), moved_seg, labels=label_list #type: ignore

            )
            measurements[disp_name]["dice"] = dice

            if warp_segmentations:
                moved_seg_nib = nib.Nifti1Image(moved_seg, affine=fixed_nib.affine)
                warped_seg_name = disp_name.replace('disp','seg')
                nib.save(
                    moved_seg_nib,
                    save_directory / "segmentations" / warped_seg_name,
                )

        # log total registration error if keypoints present
        if data.fixed_keypoints is not None:
            spacing_fix = get_spacing(fixed_nib)
            spacing_mov = get_spacing(moving_nib)
            fixed_keypoints = np.loadtxt(data.fixed_keypoints, delimiter=",")
            moving_keypoints = np.loadtxt(data.moving_keypoints, delimiter=",")

            tre_np = compute_total_registration_error(
                fixed_keypoints, moving_keypoints, l2r_disp, spacing_fix, spacing_mov
            )

            spacing_fix = torch.from_numpy(spacing_fix)
            spacing_mov = torch.from_numpy(spacing_mov)
            fixed_keypoints = torch.from_numpy(fixed_keypoints)
            moving_keypoints = torch.from_numpy(moving_keypoints)

            tre = trl(fixed_keypoints, moving_keypoints, disp_hr, spacing_fix, spacing_mov)

            measurements[disp_name]["total_registration_error"] = tre_np
            measurements[disp_name]['tre_torch'] = tre.item()

        if warp_images:
            moved_image = apply_displacement_field(disp_np, moving.detach().cpu().numpy())
            moved_nib = nib.Nifti1Image(moved_image, affine=fixed_nib.affine)
            warped_img_name = disp_name.replace('disp','img')
            nib.save(moved_nib, save_directory / "images" / warped_img_name)

        # L2R evaluation scripts expects displacements with the shape (*data_shape, 3)
        # l2r_disp = einops.rearrange(disp_np, 't h w d -> h w d t')
        if disp_format == DisplacementFormat.Numpy:
            np.savez_compressed(save_directory / "disps" / f"{disp_name}.npz", l2r_disp)
        else:
            displacement_nib = nib.Nifti1Image(l2r_disp, affine=moving_nib.affine)
            nib.save(displacement_nib, save_directory / "disps" / f"{disp_name}.nii.gz")

        torch.cuda.synchronize()

    with open(save_directory / "measurements.json", "w") as f:
        json.dump(measurements, f)

app()
