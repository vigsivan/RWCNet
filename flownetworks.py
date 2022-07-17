"""
Flownet inspired models
"""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import logging
import sys
import torchio as tio
from typing import Dict, Optional
from contextlib import nullcontext

import einops
import numpy as np
import nibabel as nib
from tqdm import tqdm, trange
import torch
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter
import typer

from common import (
    DisplacementFormat,
    MINDSSC,
    TrainType,
    adam_optimization,
    data_generator,
    random_never_ending_generator,
    randomized_pair_never_ending_generator,
    random_unpaired_never_ending_generator,
    random_unpaired_split_never_ending_generator,
    load_labels,
    load_keypoints,
    load_keypoints_np,
    torch2skimage_disp,
    tb_log,
    warp_image,
)
from differentiable_metrics import (
    MSE,
    NCC,
    MINDLoss,
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
from networks import SpatialTransformer, FlowNetCorr, VecInt, Cascade

app = typer.Typer()
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

add_bc_dim = lambda x: einops.repeat(x, "d h w -> b c d h w", b=1, c=1).float()
get_spacing = lambda x: np.sqrt(np.sum(x.affine[:3, :3] * x.affine[:3, :3], axis=0))


class ImageLoss(str, Enum):
    MutualInformation = "image_loss"
    MeanSquareError = "mse"
    NormalizedCrossCorrelation = "ncc"


def get_loss_fn(loss: ImageLoss):
    if loss == ImageLoss.MeanSquareError:
        return MSE()
    if loss == ImageLoss.NormalizedCrossCorrelation:
        return NCC()
    else:
        return MutualInformationLoss()


@dataclass
class RunModelOut:
    flow: torch.Tensor
    fixed: torch.Tensor
    moving: torch.Tensor
    moved: torch.Tensor
    total_loss: torch.Tensor
    losses_dict: Dict[str, float]


def run_flownetc(
    data,
    model,
    skip_normalize: bool,
    image_loss_weight: float,
    reg_loss_weight: float,
    dice_loss_weight: float,
    kp_loss_weight: float,
    use_labels: bool,
    use_keypoints: bool,
    with_grad: bool = True,
    device: str = "cuda",
    ) -> RunModelOut:

    context = nullcontext() if with_grad else torch.no_grad()

    with context:
        fixed_nib = nib.load(data.fixed_image)
        moving_nib = nib.load(data.moving_image)

        fixed = add_bc_dim(torch.from_numpy(fixed_nib.get_fdata())).to(device)
        moving = add_bc_dim(torch.from_numpy(moving_nib.get_fdata())).to(device)

        if not skip_normalize:
            fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min())
            moving = (moving - moving.min()) / (moving.max() - moving.min())

        flow = model(moving, fixed)
        flow = VecInt(fixed.shape[2:], nsteps=7).to(device)(flow)
        transformer = SpatialTransformer(fixed.shape[2:]).to(device)
        moved = transformer(moving, flow)

        losses_dict: Dict[str, torch.Tensor] = {}
        losses_dict["image_loss"] = image_loss_weight * MINDLoss()(
            moved, fixed
        )
        losses_dict["grad"] = reg_loss_weight * Grad()(flow)

        if use_labels:

            fixed_seg = add_bc_dim(
                torch.from_numpy(
                    np.round(nib.load(data.fixed_segmentation).get_fdata())
                )
            ).to(device)
            moving_seg = add_bc_dim(
                torch.from_numpy(
                    np.round(nib.load(data.moving_segmentation).get_fdata())
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

        loss = sum(losses_dict.values())
        assert isinstance(loss, torch.Tensor)
        losses_dict_log = {k: v.item() for k, v in losses_dict.items()}

        return RunModelOut(
            flow=flow,
            fixed=fixed,
            moving=moving,
            moved=moved,
            total_loss = loss,
            losses_dict=losses_dict_log
        )

def run_flownetcascade(
    data,
    flownet,
    cascade,
    skip_normalize: bool,
    image_loss_weight: float,
    reg_loss_weight: float,
    dice_loss_weight: float,
    kp_loss_weight: float,
    use_labels: bool,
    use_keypoints: bool,
    with_grad: bool = True,
    vector_integration_steps: int=7,
    device: str = "cuda",
    with_instance_opt: bool=False,
    apply_augmentation: bool=True,
    ) -> RunModelOut:

    context = nullcontext() if with_grad else torch.no_grad()
    aug_func =  tio.OneOf([
        tio.RandomBlur(),
        tio.RandomNoise(),
        tio.RandomGamma(),
        ])


    with context:

        if apply_augmentation:
            if not (use_labels or use_keypoints):
                aug_func = tio.Compose([tio.RandomElasticDeformation(), aug_func])

            fixed_tio = tio.ScalarImage(data.fixed_image)
            moving_tio = tio.ScalarImage(data.moving_image)

            fixed_tio, moving_tio = aug_func(fixed_tio), aug_func(moving_tio)
            fixed, moving = fixed_tio.data.unsqueeze(0), fixed_tio.data.unsqueeze(0)

            assert isinstance(fixed, torch.Tensor)
            assert isinstance(moving, torch.Tensor)

            fixed, moving = fixed.to(device), moving.to(device)

        else:
            fixed_nib = nib.load(data.fixed_image)
            moving_nib = nib.load(data.moving_image)

            fixed = add_bc_dim(torch.from_numpy(fixed_nib.get_fdata()))
            moving = add_bc_dim(torch.from_numpy(moving_nib.get_fdata()))

            fixed, moving = fixed.to(device), moving.to(device)

        if not skip_normalize:
            fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min())
            moving = (moving - moving.min()) / (moving.max() - moving.min())

        flow = flownet(moving, fixed, resize=False)

        moving_ = F.interpolate(moving, flow.shape[-3:])
        fixed_ = F.interpolate(fixed, flow.shape[-3:])

        transformer_half = SpatialTransformer(fixed_.shape[2:]).to(device)

        flow, l2s = cascade(moving_, fixed_, flow, transformer_half)
        if with_instance_opt and with_grad:
            net = adam_optimization(
                disp=flow,
                mind_fixed=MINDSSC(fixed_),
                mind_moving=MINDSSC(moving_),
                lambda_weight=1.25,
                image_shape=fixed_.shape[-3:],
                iterations=100
            )

            disp_sample = F.avg_pool3d(
                F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1
            ).permute(0, 2, 3, 4, 1)
            flow = disp_sample.permute(0, 4, 1, 2, 3)

        flow = VecInt(nsteps=vector_integration_steps, transformer=transformer_half)(flow)

        flow = flownet.refine(flownet.flow_resize(flow))
        moved = warp_image(flow, moving)

        losses_dict: Dict[str, torch.Tensor] = {}
        losses_dict["image_loss"] = image_loss_weight * MINDLoss()(
            moved.squeeze(), fixed.squeeze()
        )
        losses_dict["grad"] = reg_loss_weight * Grad()(flow)
        weights = torch.linspace(reg_loss_weight, .1, steps=len(l2s))
        l2_total = 0.
        for weight, l2 in zip(weights, l2s):
            l2_total+= l2 * weight

        assert isinstance(l2_total, torch.Tensor)
        losses_dict["l2"] = l2_total

        if use_labels:

            fixed_seg = add_bc_dim(
                torch.from_numpy(
                    np.round(nib.load(data.fixed_segmentation).get_fdata())
                )
            ).to(device)
            moving_seg = add_bc_dim(
                torch.from_numpy(
                    np.round(nib.load(data.moving_segmentation).get_fdata())
                )
            ).to(device)

            moved_seg = torch.round(
                warp_image(flow, moving_seg.float(), mode="nearest")
            )
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


        loss = sum(losses_dict.values())
        assert isinstance(loss, torch.Tensor)
        losses_dict_log = {k: v.item() for k, v in losses_dict.items()}

        return RunModelOut(
            flow=flow,
            fixed=fixed,
            moving=moving,
            moved=moved,
            total_loss = loss,
            losses_dict=losses_dict_log
        )

@app.command()
def train(
    data_json: Path,
    checkpoint_dir: Path,
    start: Optional[Path] = None,
    steps: int = 1000,
    lr: float = 3e-4,
    correlation_patch_size: int = 3,
    flownet_redir_feats: int = 32,
    feature_extractor_strides: str = "2,1,1",
    feature_extractor_feature_sizes: str = "8,32,64",
    feature_extractor_kernel_sizes: str = "7,5,5",
    skip_normalize: bool = False,
    train_paired: bool = True,
    val_paired: bool = True,
    device: str = "cuda",
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
        train_gen = random_unpaired_never_ending_generator(
            data_json, split="train", seed=42
        )

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

    starting_step = 0
    if start is not None:
        flownetc.load_state_dict(torch.load(start))
        if "step" in start.name:
            starting_step = int(start.name.split("_step")[-1].split(".")[0])

    opt = torch.optim.Adam(flownetc.parameters(), lr=lr)

    data_sample = next(train_gen)
    use_labels = data_sample.fixed_segmentation is not None
    use_keypoints = (
        data_sample.fixed_keypoints is not None and data_sample.moving_image is not None
    )

    if use_labels:
        labels = load_labels(data_json)
        logging.debug(f"Using labels. Found labels {','.join(str(i) for i in labels)}")

    print(f"Starting training from step {starting_step}")
    for step in trange(starting_step, steps + starting_step):
        data = next(train_gen)

        model_out = run_flownetc(
            data,
            flownetc,
            skip_normalize,
            image_loss_weight,
            reg_loss_weight,
            dice_loss_weight,
            kp_loss_weight,
            use_labels,
            use_keypoints,
        )

        opt.zero_grad()
        model_out.total_loss.backward()
        opt.step()

        if step % log_freq == 0:
            tb_log(
                writer,
                model_out.losses_dict,
                step=step,
                moving_fixed_moved=(model_out.moving, model_out.fixed, model_out.moved),
            )

        if step % save_freq == 0:
            torch.save(
                flownetc.state_dict(), checkpoint_dir / f"flownet_step{step}.pth",
            )

        if val_freq > 0 and step % val_freq == 0:
            flownetc.eval()

            if val_paired:
                val_gen = data_generator(data_json, split="val")
            else:
                raise NotImplementedError("Unpaired validation not implemented!")

            losses_cum_dict = defaultdict(list)
            for data in val_gen:
                model_out = run_flownetc(
                    data,
                    flownetc,
                    skip_normalize,
                    image_loss_weight,
                    reg_loss_weight,
                    dice_loss_weight,
                    kp_loss_weight,
                    use_labels,
                    use_keypoints,
                    with_grad=False,
                )
                for k,v in model_out.losses_dict.items():
                    losses_cum_dict[k].append(v)

            for k, v in losses_cum_dict.items():
                writer.add_scalar(f"val_{k}", np.mean(v).item(), global_step=step)

            flownetc = flownetc.train()

    torch.save(
        flownetc.state_dict(), checkpoint_dir / f"flownet_step{starting_step+steps}.pth"
    )


@app.command()
def train_cascade(
    flownetc_checkpoint: Path,
    data_json: Path,
    checkpoint_dir: Path,
    cascade_checkpoint: Optional[Path]=None,
    skip_normalize: bool = False,
    steps: int = 1000,
    lr: float = 3e-4,
    n_cascades: int = 6,
    freeze_corr_weights: bool=False,
    correlation_patch_size: int = 3,
    flownet_redir_feats: int = 32,
    feature_extractor_strides: str = "2,1,1",
    feature_extractor_feature_sizes: str = "8,32,64",
    feature_extractor_kernel_sizes: str = "7,5,5",
    train_type: TrainType = TrainType.Paired,
    device: str = "cuda",
    image_loss_weight: float = 100,
    dice_loss_weight: float = 0.01,
    reg_loss_weight: float = 0.01,
    kp_loss_weight: float = .1,
    log_freq: int = 5,
    save_freq: int = 100,
    val_freq: int = 0,
):
    """
    Trains a FlowNetC Model.
    """

    if train_type == TrainType.Paired:
        train_gen = random_never_ending_generator(data_json, split="train", random_switch=True, seed=42)
    elif train_type == TrainType.UnpairedSet:
        train_gen = random_unpaired_split_never_ending_generator(data_json, seed=42)
    elif train_type == TrainType.Disregard:
        train_gen = randomized_pair_never_ending_generator(data_json, split="train", seed=42)
    else:
        train_gen = random_unpaired_never_ending_generator(
            data_json, split="train", seed=42
        )

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

    checkpoint = torch.load(flownetc_checkpoint)
    flownetc.load_state_dict(checkpoint)
    if freeze_corr_weights:
        flownetc.requires_grad = False
    cascade = Cascade(N=n_cascades).to(device)

    if cascade_checkpoint is None:
        cascade_statedict = cascade.state_dict()
        # first conv layer size not going to be the same due to different number of channels
        skip_keys = ["encoder.0.0.main.weight", "encoder.0.0.main.bias"]
        for param, weight in checkpoint.items():
            if not param.startswith("unet"):
                continue
            key = param[5:]  # skip unet.
            if key in skip_keys:
                continue
            for k in cascade_statedict.keys():
                if key in k:
                    cascade_statedict[k] = weight

    else:
        cascade_statedict = torch.load(cascade_checkpoint)
        
    cascade.load_state_dict(cascade_statedict)

    opt = torch.optim.Adam(flownetc.parameters(), lr=lr)

    for step in trange(steps):
        data = next(train_gen)

        train_use_labels = (
            data.fixed_segmentation is not None
            and data.moving_segmentation is not None
        )
        train_use_keypoints = (
            data.fixed_keypoints is not None and data.moving_image is not None
        )

        model_out = run_flownetcascade(
            data,
            flownetc,
            cascade,
            skip_normalize,
            image_loss_weight,
            reg_loss_weight,
            dice_loss_weight,
            kp_loss_weight,
            train_use_labels,
            train_use_keypoints,
        )

        opt.zero_grad()
        model_out.total_loss.backward()
        opt.step()

        if step % log_freq == 0:
            tb_log(
                writer,
                model_out.losses_dict,
                step=step,
                moving_fixed_moved=(model_out.moving, model_out.fixed, model_out.moved),
            )

        if step % save_freq == 0:
            torch.save(
                flownetc.state_dict(), checkpoint_dir / f"flownetc_pre_step{step}.pth"
            )
            torch.save(
                cascade.state_dict(), checkpoint_dir / f"cascade_step{step}.pth",
            )

        if val_freq > 0 and step % val_freq == 0:
            flownetc.eval()
            val_gen = data_generator(data_json, split="val")
            losses_cum_dict = defaultdict(list)
            for data in val_gen:
                val_use_labels = (
                    data.fixed_segmentation is not None
                    and data.moving_segmentation is not None
                )
                val_use_keypoints = (
                    data.fixed_keypoints is not None and data.moving_image is not None
                )

                model_out = run_flownetcascade(
                    data,
                    flownetc,
                    cascade,
                    skip_normalize,
                    image_loss_weight,
                    reg_loss_weight,
                    dice_loss_weight,
                    kp_loss_weight,
                    val_use_labels,
                    val_use_keypoints,
                    with_grad=False,
                )
                for k,v in model_out.losses_dict.items():
                    losses_cum_dict[k].append(v)

            for k, v in losses_cum_dict.items():
                writer.add_scalar(f"val_{k}", np.mean(v).item(), global_step=step)

            flownetc = flownetc.train()

    torch.save(flownetc.state_dict(), checkpoint_dir / f"flownetc_pre_step{steps}.pth")
    torch.save(cascade.state_dict(), checkpoint_dir / f"cascade_step{steps}.pth")


@app.command()
def eval_cascade(
    flownetc_checkpoint: Path,
    cascades_checkpoint: Path,
    data_json: Path,
    savedir: Path,
    skip_normalize: bool = False,
    n_cascades: int = 6,
    correlation_patch_size: int = 3,
    flownet_redir_feats: int = 32,
    feature_extractor_strides: str = "2,1,1",
    feature_extractor_feature_sizes: str = "8,32,64",
    feature_extractor_kernel_sizes: str = "7,5,5",
    use_l2r_naming: bool = True,
    disp_format: DisplacementFormat = DisplacementFormat.Nifti,
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
    flownetc.load_state_dict(torch.load(flownetc_checkpoint))
    flownetc = flownetc.to(device).eval()

    cascade = Cascade(N=n_cascades)
    cascade.load_state_dict(torch.load(cascades_checkpoint))
    cascade = cascade.to(device).eval()

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

        model_out = run_flownetcascade(
            data,
            flownetc,
            cascade,
            skip_normalize,
            image_loss_weight=1,
            reg_loss_weight=1,
            dice_loss_weight=1,
            kp_loss_weight=1,
            use_labels=use_labels,
            use_keypoints=use_keypoints,
            with_grad=False
        )

        if save_images:
            moved = warp_image(model_out.flow, model_out.moving)
            moved_image_name = (
                data.moving_image.name.split(".")[0]
                + "_warped."
                + ".".join(data.moving_image.name.split(".")[1:])
            )
            warped_nib = nib.Nifti1Image(
                moved.detach().cpu().numpy().squeeze(), affine=fixed_nib.affine
            )
            nib.save(warped_nib, savedir / "images" / moved_image_name)

        if use_l2r_naming:
            disp_name = f"disp_{data.fixed_image.name[-16:-12]}_{data.moving_image.name[-16:-12]}"
        else:
            disp_name = f"disp_{data.fixed_image.name.split('.')[0]}_{data.moving_image.name.split('.')[0]}"

        disp_np = torch2skimage_disp(model_out.flow)

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
                    np.round(nib.load(data.moving_segmentation).get_fdata())
                )
            ).to(device)

            moved_seg = (
                torch.round(warp_image(model_out.flow, moving_seg.float(), mode="nearest"))
                .detach()
                .cpu()
            )

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

        if disp_format == DisplacementFormat.Numpy:
            np.savez_compressed(savedir / "disps" / f"{disp_name}.npz", disp_np)
        else:
            displacement_nib = nib.Nifti1Image(disp_np, affine=moving_nib.affine)
            nib.save(displacement_nib, savedir / "disps" / f"{disp_name}.nii.gz")

    with open(savedir / "measurements.json", "w") as f:
        json.dump(measurements, f)


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
    skip_normalize: bool = False,
    use_l2r_naming: bool = True,
    disp_format: DisplacementFormat = DisplacementFormat.Nifti,
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

        model_out = run_flownetc(
            data,
            flownetc,
            skip_normalize,
            image_loss_weight=1,
            reg_loss_weight=1,
            dice_loss_weight=1,
            kp_loss_weight=1,
            use_labels=use_labels,
            use_keypoints=use_keypoints,
            with_grad=False
        )

        if save_images:
            moved_image_name = (
                data.moving_image.name.split(".")[0]
                + "_warped."
                + ".".join(data.moving_image.name.split(".")[1:])
            )
            warped_nib = nib.Nifti1Image(
                model_out.moved.detach().cpu().numpy().squeeze(), affine=fixed_nib.affine
            )
            nib.save(warped_nib, savedir / "images" / moved_image_name)

        if use_l2r_naming:
            disp_name = f"disp_{data.fixed_image.name[-16:-12]}_{data.moving_image.name[-16:-12]}"
        else:
            disp_name = f"disp_{data.fixed_image.name.split('.')[0]}_{data.moving_image.name.split('.')[0]}"

        disp_np = torch2skimage_disp(model_out.flow)

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
                    np.round(nib.load(data.moving_segmentation).get_fdata())
                )
            ).to(device)

            moved_seg = (
                torch.round(warp_image(model_out.flow, moving_seg.float(), mode="nearest"))
                .detach()
                .cpu()
            )

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

        if disp_format == DisplacementFormat.Numpy:
            np.savez_compressed(savedir / "disps" / f"{disp_name}.npz", disp_np)
        else:
            displacement_nib = nib.Nifti1Image(disp_np, affine=moving_nib.affine)
            nib.save(displacement_nib, savedir / "disps" / f"{disp_name}.nii.gz")

    with open(savedir / "measurements.json", "w") as f:
        json.dump(measurements, f)


app()
