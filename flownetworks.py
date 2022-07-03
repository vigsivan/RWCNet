"""
Flownet inspired models
"""

from collections import defaultdict
from enum import Enum
import json
from pathlib import Path
import logging
import sys
from typing import Dict

import einops
import numpy as np
import nibabel as nib
from tqdm import tqdm, trange
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import typer

from common import (
    data_generator,
    random_never_ending_generator,
    random_unpaired_never_ending_generator,
    load_labels,
    load_keypoints,
    load_keypoints_np,
    torch2skimage_disp,
    tb_log,
)
from differentiable_metrics import (
    MSE,
    NCC,
    MutualInformationLoss,
    DiceLoss,
    Grad,
    TotalRegistrationLoss,
)
from metrics import (
    compute_dice,
    compute_hd95,
    compute_log_jacobian_determinant_standard_deviation,
    compute_total_registration_error,
)
from networks import SpatialTransformer, FlowNetCorr, VecInt

app = typer.Typer()
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

add_bc_dim = lambda x: einops.repeat(x, "d h w -> b c d h w", b=1, c=1).float()
get_spacing = lambda x: np.sqrt(np.sum(x.affine[:3, :3] * x.affine[:3, :3], axis=0))


class ImageLoss(str, Enum):
    MutualInformation = "mi"
    MeanSquareError = "mse"
    NormalizedCrossCorrelation = "ncc"


def get_loss_fn(loss: ImageLoss):
    if loss == ImageLoss.MeanSquareError:
        return MSE()
    if loss == ImageLoss.NormalizedCrossCorrelation:
        return NCC()
    else:
        return MutualInformationLoss()


@app.command()
def train(
    data_json: Path,
    checkpoint_dir: Path,
    steps: int = 1000,
    lr: float = 3e-4,
    correlation_patch_size: int = 3,
    flownet_redir_feats: int = 32,
    feature_extractor_strides: str = "2,1,1",
    feature_extractor_feature_sizes: str = "8,32,64",
    feature_extractor_kernel_sizes: str = "7,5,5",
    enforce_inverse_consistency: bool=False, # TODO
    train_paired: bool=True,
    val_paired: bool=True,
    device: str = "cuda",
    image_loss: ImageLoss = ImageLoss.MutualInformation,
    image_loss_weight: float = 1,
    dice_loss_weight: float = 1.0,
    reg_loss_weight: float = 0.01,
    kp_loss_weight: float = 1,
    log_freq: int = 5,
    save_freq: int = 100,
    val_freq: int = 0,
):
    """
    Trains a FlowNetC Model.

    Parameters
    ----------
    data_json: Path
    checkpoint_dir: Path
        This is where checkpoints are saved
    steps: int
        Total number of training steps. Default=1000
    lr: float
        Adam Optimizer learning rate. Default=3e-4
    feature_extractor_strides: str
        Comma-separated string. If not provided, default is 2,1,1
    feature_extractor_feature_sizes: str
        Comma-separated string. If not provided, default is 8,32,64
    feature_extractor_kernel_sizes: str
        Comma-separated string. If not provided, default is 7,5,5
    enforce_inverse_consistency: bool
        Default=False, # TODO
    train_paired: bool
        Default=True,
    val_paired: bool
        Default=True,
    device: str = 
        Default="cuda",
    image_loss: ImageLoss 
        Default= ImageLoss.MutualInformation,
    image_loss_weight: float
        Image similarity loss weight. Default=1.
    dice_loss_weight
        Dice loss weight. Note, this only applies if labels present in the dataset. Default=1.
    reg_loss_weight
        Regularization loss weight. Default=.01
    kp_loss_weight
        Keypoints loss weight. Note, only applies when keypoints provided. Default=1.
    log_freq: int
        Frequency at which to write losses to tensorboard. Default=5
    save_freq: int
        Frequency at which to save models. Default=100
    val_feq: int
        Frequency at which to evaluate model against validation. Default=0 (never).
    """
    if train_paired:
        train_gen = random_never_ending_generator(data_json, split="train", seed=42)
    else:
        train_gen = random_unpaired_never_ending_generator(data_json, split="train", seed=42)

    checkpoint_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=checkpoint_dir)

    flownet_kwargs = {
        "feature_extractor_strides": [
            int(i.strip()) for i in feature_extractor_strides.split(",")
        ],
        "feature_extractor_feature_sizes": [
            int(i.strip()) for i in feature_extractor_feature_sizes.split(",")
        ],
        "feature_extractor_kernel_sizes": [
            int(i.strip()) for i in feature_extractor_kernel_sizes.split(",")
        ],
    }

    assert (
        len(set(len(v) for v in flownet_kwargs.values())) == 1
    ), "Feature extractor list params must all have the same size"

    flownetc = FlowNetCorr(
        correlation_patch_size=correlation_patch_size,
        redir_feats=flownet_redir_feats,
        **flownet_kwargs,
    ).to(device)

    opt = torch.optim.Adam(flownetc.parameters(), lr=lr)

    image_loss_fn = get_loss_fn(image_loss)
    grad_loss_fn = Grad()

    data_sample = next(train_gen)
    use_labels = data_sample.fixed_segmentation is not None
    use_keypoints = (
        data_sample.fixed_keypoints is not None and data_sample.moving_image is not None
    )

    if use_labels:
        labels = load_labels(data_json)
        logging.debug(f"Using labels. Found labels {','.join(str(i) for i in labels)}")

    for step in trange(steps):
        data = next(train_gen)

        fixed_nib = nib.load(data.fixed_image)
        moving_nib = nib.load(data.moving_image)

        fixed = add_bc_dim(torch.from_numpy(fixed_nib.get_fdata())).to(device)
        moving = add_bc_dim(torch.from_numpy(moving_nib.get_fdata())).to(device)

        flow = flownetc(moving, fixed)
        flow = VecInt(fixed.shape[2:], nsteps=7).to(device)(flow)
        transformer = SpatialTransformer(fixed.shape[2:]).to(device)
        moved = transformer(moving, flow)

        losses_dict: Dict[str, torch.Tensor] = {}
        losses_dict["mi"] = image_loss_weight * image_loss_fn(moved, fixed)
        losses_dict["grad"] = reg_loss_weight * grad_loss_fn(flow)

        if use_labels:

            fixed_seg = add_bc_dim(
                torch.from_numpy(
                    np.round(nib.load(data.fixed_segmentation).get_fdata())
                )
            ).to(device)
            moving_seg = add_bc_dim(
                torch.from_numpy(
                    np.round(nib.load(data.fixed_segmentation).get_fdata())
                )
            ).to(device)

            moved_seg = torch.round(transformer(moving_seg.float(), flow))
            losses_dict["dice"] = dice_loss_weight * DiceLoss()(
                fixed_seg.squeeze(), moving_seg.squeeze(), moved_seg.squeeze()
            )

        if use_keypoints:
            assert (
                data.fixed_keypoints is not None and data.moving_keypoints is not None
            )

            fixed_kps = load_keypoints(data.fixed_keypoints)
            moving_kps = load_keypoints(data.moving_keypoints)
            losses_dict["keypoints"] = kp_loss_weight * TotalRegistrationLoss()(
                fixed_landmarks=fixed_kps,
                moving_landmarks=moving_kps,
                displacement_field=flow,
                fixed_spacing=torch.Tensor(get_spacing(fixed_nib)),
                moving_spacing=torch.Tensor(get_spacing(moving_nib)),
            )

        opt.zero_grad()
        loss = sum(losses_dict.values())
        assert isinstance(loss, torch.Tensor)
        loss.backward()
        opt.step()

        if step % log_freq == 0:
            tb_log(
                writer,
                losses_dict,
                step=step,
                moving_fixed_moved=(moving, fixed, moved),
            )

        if step % save_freq == 0:
            torch.save(
                flownetc.state_dict(), checkpoint_dir / f"flownet_step{step}.pth",
            )

        if val_freq > 0 and step % val_freq == 0:
            flownetc.eval()
            with torch.no_grad():
                if val_paired:
                    val_gen = data_generator(data_json, split="val")
                else:
                    raise NotImplementedError("Unpaired validation not implemented!")
                losses_cum_dict = defaultdict(list)
                for data in val_gen:

                    fixed_nib = nib.load(data.fixed_image)
                    moving_nib = nib.load(data.moving_image)

                    fixed = add_bc_dim(torch.from_numpy(fixed_nib.get_fdata())).to(device)
                    moving = add_bc_dim(torch.from_numpy(moving_nib.get_fdata())).to(device)

                    flow = flownetc(moving, fixed)
                    flow = VecInt(fixed.shape[2:], nsteps=7).to(device)(flow)
                    transformer = SpatialTransformer(fixed.shape[2:]).to(device)
                    moved = transformer(moving, flow)

                    losses_cum_dict["mi"].append(
                        (image_loss_weight * image_loss_fn(moved, fixed))
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    losses_cum_dict["grad"].append(
                        (reg_loss_weight * grad_loss_fn(flow)).detach().cpu().numpy()
                    )

                    if use_labels:

                        fixed_seg = add_bc_dim(
                            torch.from_numpy(
                                np.round(nib.load(data.fixed_segmentation).get_fdata())
                            )
                        ).to(device)
                        moving_seg = add_bc_dim(
                            torch.from_numpy(
                                np.round(nib.load(data.fixed_segmentation).get_fdata())
                            )
                        ).to(device)

                        moved_seg = torch.round(transformer(moving_seg.float(), flow, mode="nearest"))
                        losses_cum_dict["dice"].append(
                            dice_loss_weight
                            * DiceLoss()(
                                fixed_seg.squeeze(),
                                moving_seg.squeeze(),
                                moved_seg.squeeze(),
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )

                    if use_keypoints:
                        assert (
                            data.fixed_keypoints is not None
                            and data.moving_keypoints is not None
                        )

                        fixed_kps = load_keypoints(data.fixed_keypoints)
                        moving_kps = load_keypoints(data.moving_keypoints)
                        losses_cum_dict["keypoints"].append(
                            kp_loss_weight
                            * TotalRegistrationLoss()(
                                fixed_landmarks=fixed_kps,
                                moving_landmarks=moving_kps,
                                displacement_field=flow,
                                fixed_spacing=torch.Tensor(get_spacing(fixed_nib)),
                                moving_spacing=torch.Tensor(get_spacing(moving_nib)),
                            )
                        )
                for k, v in losses_cum_dict.items():
                    writer.add_scalar(f"val_{k}", np.mean(v).item(), global_step=step)

                del losses_cum_dict
                flownetc = flownetc.train()

    torch.save(flownetc.state_dict(), checkpoint_dir / f"flownet_step{steps}.pth")


@app.command()
def eval(
    checkpoint: Path,
    data_json: Path,
    savedir: Path,
    correlation_patch_size: int = 3,
    flownet_redir_feats: int = 32,
    feature_extractor_strides: str = "2,1,1",
    feature_extractor_feature_sizes: str = "8,32,64",
    feature_extractor_kernel_sizes: str = "7,5,5",
    diffeomorphic: bool=False,
    device: str = "cuda",
    save_images: bool = True,
):
    """
    Evaluates a flownet model.

    Measurements saved to savedir/measurements.json.
    Skimage compatible displacements saved to savedir/disps.

    Parameters
    ----------
    checkpoint: Path
    data_json: Path
    save_dir: Path
    device: str
        One of "cpu", "cuda". Default="cuda"
    save_images: bool
        Default=False.
    """
    savedir.mkdir(exist_ok=True)
    flownet_kwargs = {
        "feature_extractor_strides": [
            int(i.strip()) for i in feature_extractor_strides.split(",")
        ],
        "feature_extractor_feature_sizes": [
            int(i.strip()) for i in feature_extractor_feature_sizes.split(",")
        ],
        "feature_extractor_kernel_sizes": [
            int(i.strip()) for i in feature_extractor_kernel_sizes.split(",")
        ],
    }

    assert (
        len(set(len(v) for v in flownet_kwargs.values())) == 1
    ), "Feature extractor params must all have the same size"

    flownetc = FlowNetCorr(
        correlation_patch_size=correlation_patch_size,
        redir_feats=flownet_redir_feats,
        **flownet_kwargs,
    )
    flownetc.load_state_dict(torch.load(checkpoint))
    flownetc = flownetc.to(device).eval()
    gen = data_generator(data_json, split="val")
    labels = load_labels(data_json)

    measurements = defaultdict(dict)
    (savedir / "disps").mkdir(exist_ok=True)
    if save_images:
        (savedir / "images").mkdir(exist_ok=True)

    for data in tqdm(gen):
        use_labels = data.fixed_segmentation is not None
        use_keypoints = data.fixed_keypoints is not None

        fixed_nib = nib.load(data.fixed_image)
        moving_nib = nib.load(data.moving_image)

        fixed = add_bc_dim(torch.from_numpy(fixed_nib.get_fdata())).to(device)
        moving = add_bc_dim(torch.from_numpy(moving_nib.get_fdata())).to(device)

        flow = flownetc(moving, fixed)
        if diffeomorphic:
            flow = VecInt(fixed.shape[2:], nsteps=7).to(device)(flow)
        transformer = SpatialTransformer(fixed.shape[2:]).to(device)

        if save_images:
            moved = transformer(moving, flow)
            moved_image_name = (
                data.moving_image.name.split(".")[0]
                + "_warped."
                + ".".join(data.moving_image.name.split(".")[1:])
            )
            warped_nib = nib.Nifti1Image(
                moved.detach().cpu().numpy().squeeze(), affine=fixed_nib.affine
            )
            nib.save(warped_nib, savedir / "images" / moved_image_name)

        disp_name = f"{data.moving_image.name.split('.')[0]}2{data.fixed_image.name.split('.')[0]}"
        disp_np = torch2skimage_disp(flow)
        np.savez_compressed(savedir / "disps" / disp_name, disp_np)

        measurements[disp_name][
            "sdlogj"
        ] = compute_log_jacobian_determinant_standard_deviation(disp_np)

        if use_labels:

            fixed_seg = add_bc_dim(
                torch.from_numpy(
                    np.round(nib.load(data.fixed_segmentation).get_fdata())
                )
            )
            moving_seg = add_bc_dim(
                torch.from_numpy(
                    np.round(nib.load(data.fixed_segmentation).get_fdata())
                )
            ).to(device)

            moved_seg = torch.round(transformer(moving_seg.float(), flow, mode="nearest")).detach().cpu()

            fixed_seg, moving_seg, moved_seg = (
                fixed_seg.numpy(),
                moving_seg.detach().cpu().numpy(),
                moved_seg.numpy(),
            )

            measurements[disp_name]["dice"] = compute_dice(
                fixed_seg, moving_seg, moved_seg, labels=labels
            )
            measurements[disp_name]["hd95"] = compute_hd95(
                fixed_seg, moving_seg, moved_seg, labels
            )

        if use_keypoints:
            tre = compute_total_registration_error(
                fix_lms=load_keypoints_np(data.fixed_keypoints),
                mov_lms=load_keypoints_np(data.moving_keypoints),
                disp=disp_np,
                spacing_fix=get_spacing(fixed_nib),
                spacing_mov=get_spacing(moving_nib),
            )
            measurements[disp_name]["tre"] = tre

    with open(savedir / "measurements.json", "w") as f:
        json.dump(measurements, f)


app()