"""
Flownet inspired models
"""

from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

from common import correlate_grad
from networks import default_unet_features, Unet3D, ResizeTransform, VecInt

__all__ = ["FlowNetCorr"]


class FeatureExtractor(nn.Module):
    """
    Feature Extractor for flownet model
    """

    def __init__(
        self,
        infeats: int,
        feature_sizes: List[int],
        kernel_sizes: List[int],
        strides: List[int],
    ):
        super().__init__()
        if len(feature_sizes) != len(kernel_sizes):
            raise ValueError("Features and kernels list must be the same size")

        convs = []
        prev_feat = infeats
        for feature_size, kernel_size, stride in zip(
            feature_sizes, kernel_sizes, strides
        ):
            convs.append(_conv_bn_lrelu(prev_feat, feature_size, kernel_size, stride))
            prev_feat = feature_size
        # self.convs = nn.Sequential(*convs)
        self.convs = nn.ModuleList(convs)
        self.strides = strides

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = image
        for conv, stride in zip(self.convs, self.strides):
            og_shape = x.shape[2:]
            x = conv(x)
            padding = _compute_pad(og_shape, x.shape[2:], stride)
            x = F.pad(x, padding)
        return x


class FlowNetCorr(nn.Module):
    """
    Implements FlowNetCorr
    """

    def __init__(
        self,
        correlation_patch_size: int = 3,
        infeats: int = 1,
        feature_extractor_feature_sizes: List[int] = [8, 32, 64],
        feature_extractor_kernel_sizes: List[int] = [7, 5, 5],
        feature_extractor_strides: List[int] = [2, 1, 1],
        redir_feats: int = 32,
    ):
        super().__init__()
        self.correlation_patch_size = correlation_patch_size
        self.feature_extractor = FeatureExtractor(  # (*fex_args, **fex_kwargs)
            infeats=infeats,
            feature_sizes=feature_extractor_feature_sizes,
            kernel_sizes=feature_extractor_kernel_sizes,
            strides=feature_extractor_strides,
        )
        self.conv_redir = _conv_bn_lrelu(
            infeats=feature_extractor_feature_sizes[-1],
            outfeats=redir_feats,
            kernel_size=1,
            stride=1,
        )
        # NOTE: this is not strictly like flownet because we just pop in a unet
        corr_out_feat = (1 + (correlation_patch_size * 2)) ** 3 + redir_feats
        self.unet = Unet3D(infeats=corr_out_feat)
        self.flow = nn.Conv3d(
            default_unet_features()[-1][-1], 3, kernel_size=3, padding=1
        )

        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        resize_factor = 1 / np.prod(feature_extractor_strides).item()
        self.flow_resize = ResizeTransform(resize_factor, ndims=3)
        self.refine = RefinementModule()

    def forward(self, moving_image: torch.Tensor, fixed_image: torch.Tensor):
        feat_mov = self.feature_extractor(moving_image)
        feat_fix = self.feature_extractor(fixed_image)
        corr = correlate_grad(
            feat_fix,
            feat_mov,
            self.correlation_patch_size,
            grid_sp=1,
            image_shape=feat_mov.shape[2:],
        ).unsqueeze(0)
        redir = self.conv_redir(feat_mov)
        unet_in = torch.cat([corr, redir], dim=1)
        unet_out = self.unet(unet_in)
        flow = self.flow(unet_out)
        flow = self.flow_resize(flow)
        flow = self.refine(flow)

        return flow


class RefinementModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels=3, out_channels=3, kernel_size=1, padding="same"
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


def _conv_bn_lrelu(
    infeats: int, outfeats: int, kernel_size: int, stride: int
) -> nn.Module:
    padding = "same" if stride == 1 else 0
    return nn.Sequential(
        nn.Conv3d(
            in_channels=infeats,
            out_channels=outfeats,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm3d(outfeats),
        nn.LeakyReLU(),
    )


def _compute_pad(
    original_shape: Tuple[int, ...], actual_shape: Tuple[int, ...], stride: int,
) -> Tuple[int, ...]:
    padding = [0] * 2 * len(original_shape)
    for i, (ogs, acs) in enumerate(zip(original_shape, actual_shape)):
        expected = ogs // stride
        if expected > acs:
            half = (expected - acs) // 2
            rem = expected - acs - half
            padding[2 * i] = half
            padding[(2 * i) + 1] = rem

    # NOTE: pytorch accepts padding in reverse order
    return tuple(padding[::-1])


if __name__ == "__main__":
    from collections import defaultdict
    import json
    from pathlib import Path
    import logging
    import sys
    from typing import Dict

    import einops
    import nibabel as nib
    from tqdm import tqdm, trange
    from torch.utils.tensorboard.writer import SummaryWriter
    import typer

    from common import (
        data_generator,
        random_never_ending_generator,
        load_labels,
        load_keypoints,
        torch2skimage_disp,
        tb_log,
    )
    from differentiable_metrics import (
        MutualInformationLoss,
        DiceLoss,
        Grad,
        TotalRegistrationLoss,
    )
    from metrics import compute_dice, compute_total_registation_error
    from networks import SpatialTransformer

    app = typer.Typer()
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    to5d = lambda x: einops.repeat(x, "d h w -> b c d h w", b=1, c=1).float()
    get_spacing = lambda x: np.sqrt(np.sum(x.affine[:3, :3] * x.affine[:3, :3], axis=0))

    @app.command()
    def train(
        data_json: Path,
        checkpoint_dir: Path,
        steps: int = 1000,
        lr: float = 3e-4,
        feature_extractor_strides: str = "2,1,1",
        feature_extractor_feature_sizes: str = "8,32,64",
        feature_extractor_kernel_sizes: str = "7,5,5",
        device: str = "cuda",
        mi_loss_weight: float = 1,
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
        mi_loss_weight: float
            Mutual information loss weight. Default=1.
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
        train_gen = random_never_ending_generator(data_json, split="train")
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
        ), "Feature extractor params must all have the same size"

        flownetc = FlowNetCorr(**flownet_kwargs).to(device)
        opt = torch.optim.Adam(flownetc.parameters(), lr=lr)

        mi_loss_fn = MutualInformationLoss()
        grad_loss_fn = Grad()

        data_sample = next(train_gen)
        use_labels = data_sample.fixed_segmentation is not None
        use_keypoints = data_sample.fixed_keypoints is not None and data_sample.moving_image is not None

        if use_labels:
            labels = load_labels(data_json)
            logging.debug(
                f"Using labels. Found labels {','.join(str(i) for i in labels)}"
            )

        for step in trange(steps):
            data = next(train_gen)

            fixed_nib = nib.load(data.fixed_image)
            moving_nib = nib.load(data.moving_image)


            fixed = to5d(torch.from_numpy(fixed_nib.get_fdata())).to(device)
            moving = to5d(torch.from_numpy(moving_nib.get_fdata())).to(device)

            flow = flownetc(moving, fixed)
            flow = VecInt(fixed.shape[2:], nsteps=7).to(device)(flow)
            transformer = SpatialTransformer(fixed.shape[2:]).to(device)
            moved = transformer(moving, flow)

            losses_dict: Dict[str, torch.Tensor] = {}
            losses_dict["mi"] = mi_loss_weight * mi_loss_fn(moved, fixed)
            losses_dict["grad"] = reg_loss_weight * grad_loss_fn(flow)

            if use_labels:

                fixed_seg = to5d(
                    torch.from_numpy(
                        np.round(nib.load(data.fixed_segmentation).get_fdata())
                    )
                ).to(device)
                moving_seg = to5d(
                    torch.from_numpy(
                        np.round(nib.load(data.fixed_segmentation).get_fdata())
                    )
                ).to(device)

                moved_seg = torch.round(transformer(moving_seg.float(), flow))
                losses_dict["dice"] = dice_loss_weight * DiceLoss()(
                    fixed_seg.squeeze(), moving_seg.squeeze(), moved_seg.squeeze()
                )

            if use_keypoints:
                assert (data.fixed_keypoints is not None and 
                        data.moving_keypoints is not None)

                fixed_kps = load_keypoints(data.fixed_keypoints)
                moving_kps = load_keypoints(data.moving_keypoints)
                losses_dict["keypoints"] = kp_loss_weight*TotalRegistrationLoss()(
                    fixed_landmarks=fixed_kps,
                    moving_landmarks=moving_kps,
                    displacement_field=flow,
                    fixed_spacing=torch.Tensor(get_spacing(fixed_nib)),
                    moving_spacing=torch.Tensor(get_spacing(moving_nib))
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
                    flownetc.state_dict(), checkpoint_dir / f"flownet_step{step}.pth"
                )

            if val_freq > 0 and step % val_freq == 0:
                flownetc.eval()
                with torch.no_grad():
                    val_gen = data_generator(data_json, split="val")
                    losses_cum_dict = defaultdict(list)
                    for data in val_gen:

                        fixed_nib = nib.load(data.fixed_image)
                        moving_nib = nib.load(data.moving_image)

                        fixed = to5d(torch.from_numpy(fixed_nib.get_fdata())).to(device)
                        moving = to5d(torch.from_numpy(moving_nib.get_fdata())).to(
                            device
                        )

                        flow = flownetc(moving, fixed)
                        flow = VecInt(fixed.shape[2:], nsteps=7).to(device)(flow)
                        transformer = SpatialTransformer(fixed.shape[2:]).to(device)
                        moved = transformer(moving, flow)

                        losses_cum_dict["mi"].append(
                            (mi_loss_weight * mi_loss_fn(moved, fixed))
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        losses_cum_dict["grad"].append(
                            (reg_loss_weight * grad_loss_fn(flow))
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        if use_labels:

                            fixed_seg = to5d(
                                torch.from_numpy(
                                    np.round(
                                        nib.load(data.fixed_segmentation).get_fdata()
                                    )
                                )
                            ).to(device)
                            moving_seg = to5d(
                                torch.from_numpy(
                                    np.round(
                                        nib.load(data.fixed_segmentation).get_fdata()
                                    )
                                )
                            ).to(device)

                            moved_seg = torch.round(
                                transformer(moving_seg.float(), flow)
                            )
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
                            assert (data.fixed_keypoints is not None and 
                                    data.moving_keypoints is not None)

                            fixed_kps = load_keypoints(data.fixed_keypoints)
                            moving_kps = load_keypoints(data.moving_keypoints)
                            losses_cum_dict["keypoints"].append(
                                kp_loss_weight*TotalRegistrationLoss()(
                                    fixed_landmarks=fixed_kps,
                                    moving_landmarks=moving_kps,
                                    displacement_field=flow,
                                    fixed_spacing=torch.Tensor(get_spacing(fixed_nib)),
                                    moving_spacing=torch.Tensor(get_spacing(moving_nib))
                                )
                            )
                    for k, v in losses_cum_dict.items():
                        writer.add_scalar(
                            f"val_{k}", np.mean(v).item(), global_step=step
                        )

                    del losses_cum_dict
                    flownetc = flownetc.train()

        torch.save(flownetc.state_dict(), checkpoint_dir / f"flownet_step{steps}.pth")

    @app.command()
    def eval(
        checkpoint: Path,
        data_json: Path,
        savedir: Path,
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
        flownetc = FlowNetCorr()
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

            fixed = to5d(torch.from_numpy(fixed_nib.get_fdata())).to(device)
            moving = to5d(torch.from_numpy(moving_nib.get_fdata())).to(device)

            flow = flownetc(moving, fixed)
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

            if use_labels:

                fixed_seg = to5d(
                    torch.from_numpy(
                        np.round(nib.load(data.fixed_segmentation).get_fdata())
                    )
                )
                moving_seg = to5d(
                    torch.from_numpy(
                        np.round(nib.load(data.fixed_segmentation).get_fdata())
                    )
                ).to(device)

                moved_seg = torch.round(transformer(moving_seg.float(), flow))
                moved_seg = torch.round(moved_seg).detach().cpu()

                measurements[disp_name]["dice"] = compute_dice(
                    fixed_seg.numpy(),
                    moving_seg.detach().cpu().numpy(),
                    moved_seg.numpy(),
                    labels=labels,
                )

            if use_keypoints:
                tre = compute_total_registation_error(
                    fix_lms=np.array(data.fixed_keypoints),
                    mov_lms=np.array(data.moving_keypoints),
                    disp=disp_np,
                    spacing_fix=get_spacing(fixed_nib),
                    spacing_mov=get_spacing(moving_nib),
                )
                measurements[disp_name]["tre"] = tre

        with open(savedir / "measurements.json", "w") as f:
            json.dump(measurements, f)

    app()
