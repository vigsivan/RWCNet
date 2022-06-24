"""
Trains a feature extractor
"""

from pathlib import Path
from typing import List, Optional

import einops
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchio as tio
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm, trange
import typer

from common import (
    adam_optimization_unrolled,
    adam_optimization_grad,
    correlate_grad,
    data_generator,
    displacement_permutations_grid,
    random_never_ending_generator,
    randomized_pair_never_ending_generator,
    coupled_convex_grad,
)

from differentiable_metrics import DiceLoss, MutualInformationLoss, Grad
from networks import FeatureExtractor, SpatialTransformer

app = typer.Typer()


@app.command()
def with_convexadam(
    data_json: Path,
    checkpoint_directory: Path,
    total_steps: int = int(1e3),
    steps_per_save: int = 20,
    use_labels: bool = True,
    nfeats: int = 2,
    device: str = "cuda",
    learning_rate: float = 3e-4,
    mi_loss_weight: float = 1,
    labels: List[int] = [1],
    dice_loss_weight: float = 1,
    lambda_weight: float= 0.25,
    adam_iterations: int=100,
    log_img_frequency: Optional[int] = 1,
    grid_sp: int = 2,
    disp_hw: int = 3,
    load_checkpoint: Optional[Path] = None,
    skip_normalize: bool = False,
):
    """
    Trains feature network with convexAdam
    """

    # FIXME: fix the docstring

    if device != "cuda":
        raise ValueError(
            "Only CUDA is supported, because 3D average pooling is not supported for the CPU!"
        )

    checkpoint_directory.mkdir(exist_ok=True)
    gen = random_never_ending_generator(data_json, seed=42)

    feature_net = FeatureExtractor(infeats=1, outfeats=nfeats)
    disp_mesh_t = displacement_permutations_grid(disp_hw).to(device).half()

    starting_step = 0
    if load_checkpoint is not None:
        checkpoint = torch.load(load_checkpoint)

        feature_net.load_state_dict(checkpoint["feat_net"])
        starting_step = int(load_checkpoint.name.split("_")[-1].split(".")[0])
        print(f"Starting from step {starting_step}")

    feature_net = feature_net.to(device)

    fnet_optimizer = torch.optim.Adam(feature_net.parameters(), lr=learning_rate)
    mi_loss_fn = MutualInformationLoss()
    dice_loss_fn = DiceLoss(labels)

    writer = SummaryWriter(log_dir=checkpoint_directory)

    for step in trange(starting_step, total_steps):
        data = next(gen)

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

        # NOTE: avg_pool3d is not implemented for cpu
        feat_fix_lr = F.avg_pool3d(feat_fix_, grid_sp, stride=grid_sp)
        feat_mov_lr = F.avg_pool3d(feat_mov_, grid_sp, stride=grid_sp)

        ssd = correlate_grad(feat_fix_lr, feat_mov_lr, disp_hw, grid_sp, data_shape)

        disp_soft = coupled_convex_grad(ssd, disp_mesh_t, grid_sp, data_shape)

        del ssd

        disp_lr = F.interpolate(
            disp_soft * grid_sp,
            size=tuple(s // grid_sp for s in data_shape),
            mode="trilinear",
            align_corners=False,
        )

        feat_fix_lr = F.avg_pool3d(feat_fix_, grid_sp, stride=grid_sp)
        feat_mov_lr = F.avg_pool3d(feat_mov_, grid_sp, stride=grid_sp)

        transformer = SpatialTransformer(moving_tio.spatial_shape).to(device)

        net = adam_optimization_grad(
            disp_lr=disp_lr.detach(),
            mind_fixed=feat_fix_lr,
            mind_moving=feat_mov_lr,
            lambda_weight=lambda_weight,
            image_shape=data_shape,
            grid_sp=grid_sp,
            iterations=adam_iterations,
        )
        adam_out = net[0].weight

        disp_sample = F.avg_pool3d(
            F.avg_pool3d(adam_out, 3, stride=1, padding=1), 3, stride=1, padding=1
        ).permute(0, 2, 3, 4, 1)
        fitted_grid = disp_sample.permute(0, 4, 1, 2, 3).detach()
        disp_hr = F.interpolate(
            fitted_grid * grid_sp,
            size=data_shape,
            mode="trilinear",
            align_corners=False,
        )

        disp_final = disp_hr

        moving.requires_grad = True
        moving = einops.repeat(moving, "d h w -> b c d h w", b=1, c=1).to(device)
        moved = transformer(moving, disp_final)

        mi_loss = mi_loss_weight * mi_loss_fn(fixed, moved.squeeze())
        writer.add_scalar("mi_loss", mi_loss, global_step=step)

        moved_feat = transformer(feat_mov_, disp_final)
        mi_feat_loss = mi_loss_fn(feat_fix_.squeeze(), moved_feat.squeeze())
        writer.add_scalar("mi_feat_loss", mi_feat_loss, global_step=step)

        loss = mi_loss + mi_feat_loss
        labels_available = data.fixed_segmentation is not None and data.moving_segmentation is not None

        if use_labels and labels_available:
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

            moving_seg = einops.repeat(
                moving_seg, "d h w -> b c d h w", b=1, c=1
            )
            moved_seg = transformer(moving_seg, disp_final)

            fixed_seg = torch.round(fixed_seg)
            moving_seg = torch.round(moving_seg)
            moved_seg = torch.round(moved_seg)

            dice_loss = dice_loss_weight * dice_loss_fn(
                fixed_seg, moving_seg.squeeze(), moved_seg.squeeze()
            )

            writer.add_scalar("dice_loss", dice_loss, global_step=step)

            loss += dice_loss

        if torch.isnan(loss):
            breakpoint()

        writer.add_scalar("loss", loss, global_step=step)

        if (
            log_img_frequency is not None
            and step % log_img_frequency == 0
        ):
            slice_index = feat_mov_.shape[2] // 2
            writer.add_images(
                "features",
                img_tensor=feat_mov_[0, :, slice_index, ...].unsqueeze(1),
                global_step=step,
                dataformats="nchw",
            )

            triplet = [moving.squeeze(), fixed.squeeze(), moved.squeeze()]
            writer.add_images(
                "(moving,fixed,moved)",
                img_tensor=torch.stack(triplet)[:, slice_index, ...].unsqueeze(1),
                global_step=step,
                dataformats="nchw",
            )

        loss.backward()
        fnet_optimizer.step()

        if step == total_steps:
            break

        if (step % steps_per_save) == 0:
            torch.save(
                {
                    "feat_net": feature_net.state_dict(),
                },
                checkpoint_directory/f"model_{step}.pth",
            )

    torch.save(
        {
            "feat_net": feature_net.state_dict(),
        },
        checkpoint_directory/f"model_{starting_step+total_steps}.pth",
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


app()
