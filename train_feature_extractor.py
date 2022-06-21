"""
Trains a feature extractor
"""

from pathlib import Path
from typing import Optional

import einops
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchio as tio
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import typer

from common import (
    correlate_grad,
    data_generator,
    random_never_ending_generator,
    randomized_pair_never_ending_generator,
    coupled_convex_grad,
)

from differentiable_metrics import DiceLoss, MutualInformationLoss, Grad, MINDLoss
from networks import FeatureExtractor, SpatialTransformer, VxmDense3D

app = typer.Typer()


@app.command()
def with_convexadam(
    data_json: Path,
    checkpoint_directory: Path,
    nfeats: int=12,
    device: str = "cuda",
    epochs: int = 1500,
    feature_loss_weight: float=.01,
    learning_rate: float=1e-4,
    mi_loss_weight: float=10.,
    disp_loss_weight: float=1.,
    steps_per_epoch: int = 100,
    epochs_per_save: int = 10,
    grid_sp: int = 2,
    disp_hw: int = 3,
    load_checkpoint: Optional[Path] = None,
    skip_normalize: bool = False,
):
    """
    Trains feature network with convexAdam
    """

    # FIXME: fix the docstring
    # TODO: implement labels
    if device != "cuda":
        raise ValueError("Only CUDA is supported, because average pooling is not supported for the CPU!")

    checkpoint_directory.mkdir(exist_ok=True)
    gen = randomized_pair_never_ending_generator(data_json, seed=42)
    feature_net = FeatureExtractor(infeats=1, outfeats=nfeats)
    starting_epoch = 0
    if load_checkpoint is not None:
        feature_net.load_state_dict(torch.load(load_checkpoint))
        starting_epoch = int(load_checkpoint.name.split('_')[-1].split('.')[0])
        print(f"Starting from epoch {starting_epoch}")
    feature_net = feature_net.to(device)

    fnet_optimizer = torch.optim.Adam(feature_net.parameters(), lr=learning_rate)

    mi_loss_fn = MINDLoss() #MutualInformationLoss()

    writer = SummaryWriter(log_dir=checkpoint_directory)

    for epoch in tqdm(range(starting_epoch, starting_epoch+epochs)):
        for step, data in enumerate(gen, start=1):

            fnet_optimizer.zero_grad()

            fixed_tio = tio.ScalarImage(data.fixed_image)
            moving_tio = tio.ScalarImage(data.moving_image)

            data_shape = fixed_tio.spatial_shape

            if not skip_normalize:
                fixed_tio = tio.RescaleIntensity()(fixed_tio)
                moving_tio = tio.RescaleIntensity()(moving_tio)
                # Make the typer-checker happy
                assert isinstance(fixed_tio, tio.ScalarImage)
                assert isinstance(moving_tio, tio.ScalarImage)

            # Squeeze is needed because tio automatically adds a channel dimension
            fixed = fixed_tio.data.float().squeeze().to(device)
            moving = moving_tio.data.float().squeeze().to(device)

            feat_fix_ = feature_net(fixed.unsqueeze(0).unsqueeze(0).to(device))
            feat_mov_ = feature_net(moving.unsqueeze(0).unsqueeze(0).to(device))

            avg_fix = torch.sum(feat_fix_, dim=1).unsqueeze(1)/feat_mov_.shape[1]
            avg_mov = torch.sum(feat_mov_, dim=1).unsqueeze(1)/feat_mov_.shape[1]

            avg_fix_norm = ((avg_fix-avg_fix.min())/(avg_fix.max()-avg_fix.min()))
            avg_mov_norm = ((avg_mov-avg_mov.min())/(avg_mov.max()-avg_mov.min()))

            avg_fix_one_hot = torch.round(avg_fix_norm)
            avg_mov_one_hot = torch.round(avg_mov_norm)

            # NOTE: avg_pool3d is not implemented for cpu
            # mind_fix_ = F.avg_pool3d(feat_fix_, grid_sp, stride=grid_sp)
            # mind_mov_ = F.avg_pool3d(feat_mov_, grid_sp, stride=grid_sp)
            mind_fix_ = F.avg_pool3d(feat_fix_, grid_sp, stride=grid_sp)
            mind_mov_ = F.avg_pool3d(feat_mov_, grid_sp, stride=grid_sp)

            ssd = correlate_grad(mind_fix_, mind_mov_, disp_hw, grid_sp, data_shape)

            # this has shape (3, disp_hw**3, 1)
            # every permutation of displacement essentially
            disp_mesh_t = (
                F.affine_grid(
                    disp_hw * torch.eye(3, 4).to(device).half().unsqueeze(0),
                    [1, 1, disp_hw * 2 + 1, disp_hw * 2 + 1, disp_hw * 2 + 1],
                    align_corners=True,
                )
                .permute(0, 4, 1, 2, 3)
                .reshape(3, -1, 1)
            )

            disp_soft = coupled_convex_grad(ssd, disp_mesh_t, grid_sp, data_shape)

            del ssd

            disp_hr = F.interpolate(
                disp_soft * grid_sp,
                size=data_shape,
                mode="trilinear",
                align_corners=False,
            )


            transformer = SpatialTransformer(moving_tio.spatial_shape).to(device)
            moving.requires_grad = True
            moving = einops.repeat(moving, "d h w -> b c d h w", b=1, c=1).to(device)
            moved = transformer(moving, disp_hr)

            mi_initial = mi_loss_fn(fixed, moving.squeeze())
            mi_final = mi_loss_fn(fixed, moved.squeeze())
            if mi_initial < mi_final:
                mi_loss = 10*mi_loss_weight*mi_final
            else:
                mi_loss = mi_loss_weight*mi_final

            writer.add_scalar(
                "mi_loss", mi_loss, global_step=(steps_per_epoch * epoch) + step
            )

            moved_feat = transformer(feat_mov_, disp_hr)
            feature_loss = feature_loss_weight * torch.linalg.norm(moved_feat- feat_fix_)

            writer.add_scalar(
                "feature_loss", feature_loss, global_step=(steps_per_epoch * epoch) + step
            )

            grad_loss = disp_loss_weight*Grad()(disp_hr)

            writer.add_scalar(
                "grad_loss", grad_loss, global_step=(steps_per_epoch * epoch) + step
            )

            if step == 0:
                assert (
                    grad_loss.requires_grad and
                    mi_loss.requires_grad
                    # feature_loss.requires_grad
                )

            loss = 10*grad_loss + feature_loss
            writer.add_scalar(
                "loss", loss, global_step=(steps_per_epoch * epoch) + step
            )

            loss.backward()
            fnet_optimizer.step()

            if step == steps_per_epoch:
                break

        if (epoch % epochs_per_save) == 0:
            torch.save(
                feature_net.state_dict(),
                checkpoint_directory / f"feat_net_{epoch}.pth",
            )

        torch.save(
            feature_net.state_dict(),
            checkpoint_directory / f"feat_net_{starting_epoch+epochs}.pth",
        )


@app.command()
def save_feats(
    data_json: Path,
    featnet_checkpoint: Path,
    save_dir: Path,
    skip_normalize: bool = False,
):
    """
    Saves images of the output features into the save_dir
    """
    save_dir.mkdir(exist_ok=True)
    gen = data_generator(data_json)
    feature_net = FeatureExtractor(1)
    feature_net.load_state_dict(torch.load(featnet_checkpoint))
    feature_net = feature_net.cuda().eval()

    for data in tqdm(gen):
        fixed_tio = tio.ScalarImage(data.fixed_image)
        moving_tio = tio.ScalarImage(data.moving_image)

        if not skip_normalize:
            fixed_tio = tio.RescaleIntensity()(fixed_tio)
            moving_tio = tio.RescaleIntensity()(moving_tio)
            # Make the typer-checker happy
            assert isinstance(fixed_tio, tio.ScalarImage)
            assert isinstance(moving_tio, tio.ScalarImage)

        # Squeeze is needed because tio automatically adds a channel dimension
        fixed = fixed_tio.data.float().squeeze().cuda()
        moving = moving_tio.data.float().squeeze().cuda()

        feat_fix_ = feature_net(fixed.unsqueeze(0).unsqueeze(0).cuda())
        feat_mov_ = feature_net(moving.unsqueeze(0).unsqueeze(0).cuda())

        for feat, fname in zip(
            (feat_mov_, feat_fix_), (data.moving_image.name, data.fixed_image.name),
        ):
            savename = save_dir / (fname.split(".")[0] + ".png")
            central_slice = feat.squeeze().shape[1] // 2
            plt.imsave(
                savename,
                feat.squeeze()[0, central_slice].detach().cpu().numpy(),
                cmap="gray",
            )


@app.command()
def with_voxelmorph(
    data_json: Path,
    checkpoint_directory: Path,
    device: str = "cuda",
    epochs: int = 1500,
    steps_per_epoch: int = 100,
    epochs_per_save: int = 100,
    load_checkpoint: Optional[Path] = None,
    voxelmorph_weights: Optional[Path] = None,
    save_vxm_weights: bool = True,
    freeze_vxm: bool = False,
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
    # FIXME
    skip_normalize: bool
        Skip normalizing the images. Functionality at the very least assumes a 
        positive intensity range. Defualt: False.
    """

    if device != "cuda":
        raise ValueError("Only CUDA is supported!")

    checkpoint_directory.mkdir(exist_ok=True)
    gen = random_never_ending_generator(data_json, seed=42)
    feature_net = FeatureExtractor(1)
    if load_checkpoint is not None:
        feature_net.load_state_dict(torch.load(load_checkpoint))
    feature_net = feature_net.to(device)

    vxm = VxmDense3D(src_feats=12, trg_feats=12).to(device)
    if voxelmorph_weights is not None:
        load_vxm_weights(vxm, voxelmorph_weights, freeze_vxm)

    fnet_optimizer = torch.optim.Adam(feature_net.parameters(), lr=1e-4)
    vxm_optimizer = torch.optim.Adam(feature_net.parameters(), lr=1e-4)

    mi_loss_fn = MutualInformationLoss()
    dice_loss_fn = DiceLoss()

    writer = SummaryWriter(log_dir=checkpoint_directory)

    for epoch in tqdm(range(epochs)):
        for step, data in enumerate(gen, start=1):

            fnet_optimizer.zero_grad()
            vxm_optimizer.zero_grad()

            fixed_tio = tio.ScalarImage(data.fixed_image)
            moving_tio = tio.ScalarImage(data.moving_image)

            # TODO: add augmentation here?

            if not skip_normalize:
                fixed_tio = tio.RescaleIntensity()(fixed_tio)
                moving_tio = tio.RescaleIntensity()(moving_tio)
                # Make the typer-checker happy
                assert isinstance(fixed_tio, tio.ScalarImage)
                assert isinstance(moving_tio, tio.ScalarImage)

            # Squeeze is needed because tio automatically adds a channel dimension
            fixed = fixed_tio.data.float().squeeze().to(device)
            moving = moving_tio.data.float().squeeze().to(device)

            feat_fix_ = feature_net(fixed.unsqueeze(0).unsqueeze(0).to(device))
            feat_mov_ = feature_net(moving.unsqueeze(0).unsqueeze(0).to(device))

            flow_field = vxm(feat_mov_, feat_fix_)
            transformer = SpatialTransformer(moving_tio.spatial_shape).to(device)
            moving = einops.repeat(moving, "d h w -> b c d h w", b=1, c=1)
            moved_image = transformer(moving, flow_field)

            loss = mi_loss_fn(fixed, moved_image.squeeze())

            if data.fixed_segmentation is not None:
                fixed_seg = (
                    tio.LabelMap(data.fixed_segmentation)
                    .data.squeeze()
                    .to(device)
                    .float()
                )
                moving_seg = (
                    tio.LabelMap(data.moving_segmentation)
                    .data.squeeze()
                    .to(device)
                    .float()
                )

                moving_seg = einops.repeat(moving_seg, "d h w -> b c d h w", b=1, c=1)
                moved_seg = transformer(moving_seg, flow_field)
                dice_loss = dice_loss_fn(fixed_seg, moved_seg.squeeze())
                loss += dice_loss

            loss.backward()
            vxm_optimizer.step()
            fnet_optimizer.step()

            writer.add_scalar(
                "loss", loss, global_step=(steps_per_epoch * epoch) + step
            )

            if step == steps_per_epoch:
                break

        if (epoch % epochs_per_save) == 0:
            torch.save(
                feature_net.state_dict(),
                checkpoint_directory / f"feat_net_{epoch}.pth",
            )
            if save_vxm_weights:
                torch.save(
                    vxm.state_dict(), checkpoint_directory / f"vxm_net_{epoch}.pth"
                )


def load_vxm_weights(
    vxm_model: torch.nn.Module, vxm_weights: Path, freeze_found_weights: bool
):
    """
    Cleanly loads voxelmorph's weights

    Parameters
    ----------
    vxm_model: nn.Module
    vxm_weights: Path
    freeze_found_weights: bool
        Freeze weights that have been found
    """
    weights = torch.load(vxm_weights)
    model_sd = vxm_model.state_dict()

    loaded_checkpoints = []

    for wname, w in weights.items():
        if wname.startswith("vxm."):
            wname = ".".join(wname.split(".")[1:])
        if wname in model_sd and model_sd[wname].shape == w.shape:
            model_sd[wname] = w
            loaded_checkpoints.append(wname)
            if freeze_found_weights:
                model_sd[wname].requires_grad = False

    if len(loaded_checkpoints) == 0:
        raise ValueError("No weights loaded!")
    vxm_model.load_state_dict(model_sd)
    return vxm_model


app()
