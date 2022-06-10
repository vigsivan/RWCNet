"""
Heavily adapted from code written by Mattias Paul
"""

from collections import defaultdict
import json
from pathlib import Path
from typing import Optional

import einops
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import typer

from common import (
    MINDSEG,
    MINDSSC,
    adam_optimization,
    apply_displacement_field,
    compute_loss,
    correlate,
    coupled_convex,
    data_generator,
    random_never_ending_generator,
)
from metrics import compute_dice, compute_total_registation_error
from networks import FeatureExtractor

app = typer.Typer()

@app.command()
def train_feature_extractor(
    data_json: Path,
    checkpoint_directory: Path,
    epochs: int=1500,
    steps_per_epoch: int=100,
    epochs_per_save: int=100,
    load_checkpoint: Optional[Path]=None,
    grid_sp: int = 2,
    disp_hw: int = 3,
    lambda_weight: float=1.25,
    skip_normalize: bool = False,
    ):
    """
    Trains a network to predict registration, but uses 

    By default this function _only_ saves scipy-compatible displacement fields
    into save_directory/disps.

    Parameters
    ----------
    data_json: Path,
        JSON containing the data paths. See definition in common.py
    checkpoint_directory: Path,
        Directory in which to save feature extractor checkpoints
    epochs: int
        Default = 1500
    steps_per_epoch: int
        Default = 100
    epochs_per_save: int
        Default = 100
    grid_sp: int = 2,
        Grid spacing. Defualt=2
    disp_hw: int = 3,
    lambda_weight: float 
        Regularization weight. Default= 1.25,
    skip_normalize: bool
        Skip normalizing the images. Functionality at the very least assumes a 
        positive intensity range. Defualt: False.
    """

    checkpoint_directory.mkdir(exist_ok=True)
    gen = random_never_ending_generator(data_json)
    feature_net = FeatureExtractor(1)
    if load_checkpoint is not None:
        feature_net.load_state_dict(torch.load(load_checkpoint))
    feature_net = feature_net.cuda()
    optimizer = torch.optim.Adam(feature_net.parameters(), lr=1e-4)

    writer = SummaryWriter(log_dir=checkpoint_directory)

    for epoch in tqdm(range(epochs)):
        for step, data in enumerate(gen, start=1):

            torch.cuda.synchronize()
            optimizer.zero_grad()

            fixed_tio = tio.ScalarImage(data.fixed_image)
            moving_tio = tio.ScalarImage(data.moving_image)

            subject = tio.Subject({"fixed": fixed_tio, "moving": moving_tio})
            transform = tio.Compose([
                tio.RandomFlip(axes=('LR')),
                tio.RandomAffine(scales=0, degrees=15) 
            ])

            transformed: tio.Subject = transform(subject)
            fixed_tio, moving_tio = transformed["fixed"], transformed["moving"]
            
            # TODO: add augmentation here?

            if not skip_normalize:
                fixed_tio = tio.RescaleIntensity()(fixed_tio)
                moving_tio = tio.RescaleIntensity()(moving_tio)
                # Make the typer-checker happy
                assert isinstance(fixed_tio, tio.ScalarImage)
                assert isinstance(moving_tio, tio.ScalarImage)

            # Squeeze is needed because tio automatically adds a channel dimension
            fixed = fixed_tio.data.float().squeeze()
            moving = moving_tio.data.float().squeeze()

            data_shape = fixed_tio.spatial_shape

            torch.cuda.synchronize()
            feat_fix_ = feature_net(fixed.unsqueeze(0).unsqueeze(0).cuda()).half()
            feat_mov_ = feature_net(moving.unsqueeze(0).unsqueeze(0).cuda()).half()

            feat_fix_ = F.avg_pool3d(feat_fix_, grid_sp, stride=grid_sp)
            feat_mov_ = F.avg_pool3d(feat_mov_, grid_sp, stride=grid_sp)
            ssd, ssd_argmin = correlate(
                feat_fix_, feat_mov_, disp_hw, grid_sp, data_shape
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

            del ssd

            torch.cuda.empty_cache()

            disp_lr = F.interpolate(
                disp_soft * grid_sp,
                size=tuple(s // 2 for s in data_shape),
                mode="trilinear",
                align_corners=False,
            )

            # extract one-hot patches
            loss = compute_loss(disp_lr, feat_fix_, feat_mov_, lambda_weight, grid_sp, data_shape)
            torch.cuda.synchronize()

            loss.backward()
            optimizer.step()

            writer.add_scalar("loss", loss, global_step=(steps_per_epoch*epoch)+step)

            if (epoch % epochs_per_save) == 0:
                torch.save(feature_net.state_dict(), checkpoint_directory/f"net_{epoch}.pth")

            if step == steps_per_epoch:
                break


@app.command()
def train_without_labels(
    data_json: Path,
    save_directory: Path,
    grid_sp: int = 2,
    disp_hw: int = 3,
    lambda_weight: float = 1.25,
    skip_normalize: bool = False,
    iterations: int = 100,
    warp_images: bool = False,
):
    """
    Performs registration without labels.

    By default this function _only_ saves scipy-compatible displacement fields
    into save_directory/disps.

    Parameters
    ----------
    data_json: Path,
        JSON containing the data paths. See definition in common.py
    save_directory: Path,
        Path where the outputs of this functin will be saved.
    grid_sp: int = 2,
        Grid spacing. Defualt=2
    disp_hw: int = 3,
    lambda_weight: float
        Regularization weight. Default = 1.25
    iterations: int = 100,
        Number of iterations of ADAM optimization. Default=100
    skip_normalize: bool
        Skip normalizing the images. Functionality at the very least assumes a 
        positive intensity range. Defualt: False.
    warp_images: bool = False,
        If True, the moving images are warped and saved in the save_directory.
        Default=False.
    """

    save_directory.mkdir(exist_ok=True)
    (save_directory / "disps").mkdir(exist_ok=True)
    measurements = defaultdict(dict)
    gen = tqdm(data_generator(data_json))

    for data in gen:

        torch.cuda.synchronize()

        fixed_tio = tio.ScalarImage(data.fixed_image)
        moving_tio = tio.ScalarImage(data.moving_image)

        if not skip_normalize:
            fixed_tio = tio.RescaleIntensity()(fixed_tio)
            moving_tio = tio.RescaleIntensity()(moving_tio)
            # Make the typer-checker happy
            assert isinstance(fixed_tio, tio.ScalarImage)
            assert isinstance(moving_tio, tio.ScalarImage)

        # Squeeze is needed because tio automatically adds a channel dimension
        fixed = fixed_tio.data.float().squeeze()
        moving = moving_tio.data.float().squeeze()

        data_shape = fixed_tio.spatial_shape

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
        disp_np = einops.rearrange(disp_np, "b c d h w -> (b c) d h w")
        disp_name = f"{data.moving_image.name}2{data.fixed_image.name}"

        if data.fixed_keypoints is not None:
            spacing_fix = np.array(fixed_tio.spacing)
            spacing_mov = np.array(moving_tio.spacing)
            fixed_keypoints = np.loadtxt(data.fixed_keypoints, delimiter=",")
            moving_keypoints = np.loadtxt(data.moving_keypoints, delimiter=",")
            tre = compute_total_registation_error(
                fixed_keypoints, moving_keypoints, disp_np, spacing_fix, spacing_mov
            )
            measurements[disp_name]["total_registration_error"] = tre

        if warp_images:
            moved_image = apply_displacement_field(disp_np, moving.numpy())
            moved_nib = nib.Nifti1Image(moved_image, affine=fixed_tio.affine)
            nib.save(moved_nib, save_directory / data.moving_image.name)

        disp_field = F.interpolate(
            disp_hr, scale_factor=0.5, mode="trilinear", align_corners=False
        )
        x1 = disp_field[0, 0, :, :, :].cpu().float().data.numpy()
        y1 = disp_field[0, 1, :, :, :].cpu().float().data.numpy()
        z1 = disp_field[0, 2, :, :, :].cpu().float().data.numpy()

        displacement = np.stack([x1, y1, z1], 0)
        np.savez_compressed(save_directory / "disps" / f"{disp_name}.npz", displacement)

        torch.cuda.synchronize()

    with open(save_directory / "measurements.json", "w") as f:
        json.dump(measurements, f)

@app.command()
def train_with_feature_extractor(
    data_json: Path,
    feature_extractor: Path,
    save_directory: Path,
    grid_sp: int = 2,
    disp_hw: int = 3,
    lambda_weight: float = 1.25,
    iterations: int = 100,
    skip_normalize: bool = False,
    warp_images: bool = False,
    warp_segmentations: bool = False,  # FIXME: don't assume segmentations by default
):
    """
    Performs registration with pre-trained feature extractor.
    """

    # TODO: fix docstring

    save_directory.mkdir(exist_ok=True)
    (save_directory / "disps").mkdir(exist_ok=True)

    if warp_images:
        warp_images_dir = save_directory / "images"
        warp_images_dir.mkdir(exist_ok=True)

    if warp_segmentations:
        warp_images_dir = save_directory / "segmentations"
        warp_images_dir.mkdir(exist_ok=True)

    gen = tqdm(data_generator(data_json))
    measurements = defaultdict(dict)

    for data in gen:

        torch.cuda.synchronize()

        fixed_tio = tio.ScalarImage(data.fixed_image)
        moving_tio = tio.ScalarImage(data.moving_image)

        if not skip_normalize:
            fixed_tio = tio.RescaleIntensity()(fixed_tio)
            moving_tio = tio.RescaleIntensity()(moving_tio)

            # Make the typer-checker happy
            assert isinstance(fixed_tio, tio.ScalarImage)
            assert isinstance(moving_tio, tio.ScalarImage)

        # Squeeze is needed because tio automatically adds a channel dimension
        fixed = fixed_tio.data.float().squeeze()
        moving = moving_tio.data.float().squeeze()

        data_shape = fixed_tio.spatial_shape

        fixed_seg = tio.LabelMap(data.fixed_segmentation).data.float().squeeze()
        moving_seg = tio.LabelMap(data.moving_segmentation).data.float().squeeze()

        feature_net = FeatureExtractor(1)
        feature_net.load_state_dict(torch.load(feature_extractor))
        feature_net = feature_net.cuda().eval()

        torch.cuda.synchronize()

        with torch.no_grad():

            feat_fix_ = feature_net(fixed.unsqueeze(0).unsqueeze(0).cuda()).half()
            feat_mov_ = feature_net(moving.unsqueeze(0).unsqueeze(0).cuda()).half()

            # FIXME: change variable names
            mind_fix_ = F.avg_pool3d(feat_fix_, grid_sp, stride=grid_sp)
            mind_mov_ = F.avg_pool3d(feat_mov_, grid_sp, stride=grid_sp)
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
            mind_fix_ = F.avg_pool3d(feat_fix_, grid_sp, stride=grid_sp)
            mind_mov_ = F.avg_pool3d(feat_mov_, grid_sp, stride=grid_sp)
        del feat_fix_, feat_mov_

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
        measurements[disp_name]["dice"] = dice

        # log total registration error if keypoints present
        if data.fixed_keypoints is not None:
            spacing_fix = np.array(fixed_tio.spacing)
            spacing_mov = np.array(moving_tio.spacing)
            fixed_keypoints = np.loadtxt(data.fixed_keypoints, delimiter=",")
            moving_keypoints = np.loadtxt(data.moving_keypoints, delimiter=",")
            tre = compute_total_registation_error(
                fixed_keypoints, moving_keypoints, disp_np, spacing_fix, spacing_mov
            )
            measurements[disp_name]["total_registration_error"] = tre

        if warp_images:
            moved_image = apply_displacement_field(disp_np, moving.numpy())
            moved_nib = nib.Nifti1Image(moved_image, affine=fixed_tio.affine)
            nib.save(moved_nib, save_directory / "images" / data.moving_image.name)

        if warp_segmentations:
            moved_seg_nib = nib.Nifti1Image(moved_seg, affine=fixed_tio.affine)
            nib.save(
                moved_seg_nib,
                save_directory / "segmentations" / data.moving_segmentation.name,
            )

        disp_field = F.interpolate(
            disp_hr, scale_factor=0.5, mode="trilinear", align_corners=False
        )
        x1 = disp_field[0, 0, :, :, :].cpu().float().data.numpy()
        y1 = disp_field[0, 1, :, :, :].cpu().float().data.numpy()
        z1 = disp_field[0, 2, :, :, :].cpu().float().data.numpy()

        displacement = np.stack([x1, y1, z1], 0)
        np.savez_compressed(save_directory / "disps" / f"{disp_name}.npz", displacement)

        torch.cuda.synchronize()

    with open(save_directory / "measurements.json", "w") as f:
        json.dump(measurements, f)


@app.command()
def train_with_labels(
    data_json: Path,
    save_directory: Path,
    grid_sp: int = 2,
    disp_hw: int = 3,
    lambda_weight: float = 1.25,
    iterations: int = 100,
    compute_mind_from_seg: bool=True,
    skip_normalize: bool = False,
    warp_images: bool = False,
    warp_segmentations: bool = False,
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

    gen = tqdm(data_generator(data_json))
    measurements = defaultdict(dict)

    for data in gen:

        torch.cuda.synchronize()

        fixed_tio = tio.ScalarImage(data.fixed_image)
        moving_tio = tio.ScalarImage(data.moving_image)


        if not skip_normalize:
            fixed_tio = tio.RescaleIntensity()(fixed_tio)
            moving_tio = tio.RescaleIntensity()(moving_tio)

            # Make the typer-checker happy
            assert isinstance(fixed_tio, tio.ScalarImage)
            assert isinstance(moving_tio, tio.ScalarImage)

        # Squeeze is needed because tio automatically adds a channel dimension
        fixed = fixed_tio.data.float().squeeze()
        moving = moving_tio.data.float().squeeze()

        data_shape = fixed_tio.spatial_shape

        fixed_seg = tio.LabelMap(data.fixed_segmentation).data.float().squeeze()
        moving_seg = tio.LabelMap(data.moving_segmentation).data.float().squeeze()


        torch.cuda.synchronize()

        with torch.no_grad():
            if compute_mind_from_seg:
                # FIXME: a lot of weird shit
                weight = 1 / (
                    torch.bincount(fixed.long().reshape(-1))
                    + torch.bincount(moving.long().reshape(-1))
                ).float().pow(0.3)
                weight /= weight.mean()

                mindssc_fix_ = MINDSEG(fixed_seg, data_shape, weight)
                mindssc_mov_ = MINDSEG(moving_seg, data_shape, weight)
            else:
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
        moved_seg = apply_displacement_field(disp_np, moving_seg.numpy())

        # Fix any problems that may have arisen due to linear interpolation
        fixed_seg[fixed_seg > 0.5] = 1
        moving_seg[moving_seg > 0.5] = 1
        moved_seg[moved_seg > 0.5] = 1

        disp_name = f"{data.moving_image.name}2{data.fixed_image.name}"
        dice = compute_dice(
            fixed_seg.numpy(), moving_seg.numpy(), moved_seg, labels=[1]
        )
        measurements[disp_name]["dice"] = dice

        # log total registration error if keypoints present
        if data.fixed_keypoints is not None:
            spacing_fix = np.array(fixed_tio.spacing)
            spacing_mov = np.array(moving_tio.spacing)
            fixed_keypoints = np.loadtxt(data.fixed_keypoints, delimiter=",")
            moving_keypoints = np.loadtxt(data.moving_keypoints, delimiter=",")
            tre = compute_total_registation_error(
                fixed_keypoints, moving_keypoints, disp_np, spacing_fix, spacing_mov
            )
            measurements[disp_name]["total_registration_error"] = tre

        if warp_images:
            moved_image = apply_displacement_field(disp_np, moving.numpy())
            moved_nib = nib.Nifti1Image(moved_image, affine=fixed_tio.affine)
            nib.save(moved_nib, save_directory / "images" / data.moving_image.name)

        if warp_segmentations:
            moved_seg_nib = nib.Nifti1Image(moved_seg, affine=fixed_tio.affine)
            nib.save(
                moved_seg_nib,
                save_directory / "segmentations" / data.moving_segmentation.name,
            )

        disp_field = F.interpolate(
            disp_hr, scale_factor=0.5, mode="trilinear", align_corners=False
        )
        x1 = disp_field[0, 0, :, :, :].cpu().float().data.numpy()
        y1 = disp_field[0, 1, :, :, :].cpu().float().data.numpy()
        z1 = disp_field[0, 2, :, :, :].cpu().float().data.numpy()

        displacement = np.stack([x1, y1, z1], 0)
        np.savez_compressed(save_directory / "disps" / f"{disp_name}.npz", displacement)

        torch.cuda.synchronize()

    with open(save_directory / "measurements.json", "w") as f:
        json.dump(measurements, f)


app()
