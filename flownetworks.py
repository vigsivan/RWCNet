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
        unet_in = torch.concat([corr, redir], dim=1)
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
    from contextlib import nullcontext
    import json
    from pathlib import Path
    import logging
    import sys
    from typing import Optional, Generator, Dict

    from tqdm import tqdm, trange
    from torch.utils.tensorboard.writer import SummaryWriter
    import torchio as tio
    import typer

    from common import (
        Data,
        LossTensors,
        data_generator,
        random_never_ending_generator,
        load_labels,
        tb_log,
    )
    from differentiable_metrics import (
        MutualInformationLoss,
        DiceLoss,
        Grad,
        TotalRegistrationLoss,
    )
    from metrics import compute_dice
    from networks import SpatialTransformer

    app = typer.Typer()
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    @app.command()
    def test_forward_pass(data_json: Path):
        gen = data_generator(data_json, split="train")
        data = next(gen)

        fixed_tio = tio.ScalarImage(data.fixed_image)
        moving_tio = tio.ScalarImage(data.moving_image)

        fixed = fixed_tio.data.unsqueeze(0)
        moving = moving_tio.data.unsqueeze(0)

        fnet = FlowNetCorr()
        flow = fnet(moving, fixed)
        SpatialTransformer(fixed.shape)(moving.unsqueeze(0).unsqueeze(0), flow)

    @app.command()
    def train(
        data_json: Path,
        checkpoint_dir: Path,
        steps: int = 1000,
        lr: float = 3e-4,
        feature_extractor_strides: Optional[str] = None,
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
        """
        train_gen = random_never_ending_generator(data_json, split="train")
        checkpoint_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir=checkpoint_dir)

        flownet_kwargs = {}
        if feature_extractor_strides is not None:
            feature_extractor_strides_list = [
                int(i.strip()) for i in feature_extractor_strides.split(",")
            ]
            flownet_kwargs["feature_extractor_strides"] = feature_extractor_strides_list

        flownetc = FlowNetCorr(**flownet_kwargs).to(device)
        opt = torch.optim.Adam(flownetc.parameters(), lr=lr)
        loss_names = ["mi", "reg"]
        loss_fns = [MutualInformationLoss(), Grad()]

        mi_loss_fn = MutualInformationLoss()
        grad_loss_fn = Grad()

        loss_weights = [mi_loss_weight, reg_loss_weight]

        if val_freq != 0:
            raise NotImplementedError(
                "Validation while model training not implemented yet"
            )  # TODO

        # TODO: training without labels
        data_sample = next(train_gen)
        use_labels = data_sample.fixed_segmentation is not None
        use_keypoints = data_sample.fixed_keypoints is not None

        if use_labels:
            labels = load_labels(data_json)
            logging.debug(
                f"Using labels. Found labels {','.join(str(i) for i in labels)}"
            )
            loss_names.append("dice")
            loss_fns.append(DiceLoss(labels))
            loss_weights.append(dice_loss_weight)

        if use_keypoints:
            logging.debug("Using keypoints.")
            loss_names.append("keypoints_loss")
            loss_fns.append(TotalRegistrationLoss())
            loss_weights.append(kp_loss_weight)

        for step in trange(steps):
            data = next(train_gen)

            fixed_tio = tio.ScalarImage(data.fixed_image)
            moving_tio = tio.ScalarImage(data.moving_image)

            fixed = fixed_tio.data.unsqueeze(0).to(device)
            moving = moving_tio.data.unsqueeze(0).to(device)

            fixed_seg = (
                tio.LabelMap(data.fixed_segmentation).data.unsqueeze(0).to(device)
            )
            moving_seg = (
                tio.LabelMap(data.fixed_segmentation).data.unsqueeze(0).to(device)
            )

            flow = flownetc(moving, fixed)
            flow = VecInt(fixed.shape[2:], nsteps=7).to(device)(flow)
            transformer = SpatialTransformer(fixed.shape[2:]).to(device)
            moved = transformer(moving, flow)

            losses_dict: Dict[str, torch.Tensor] = {}
            losses_dict["mi"] = mi_loss_weight * mi_loss_fn(moved, fixed)
            losses_dict["grad"] = reg_loss_weight * grad_loss_fn(flow)

            loss_inputs = [(moved, fixed), [flow]]

            if use_labels:
                moved_seg = torch.round(transformer(moving_seg.float(), flow))
                loss_inputs.append(
                    (fixed_seg.squeeze(), moving_seg.squeeze(), moved_seg.squeeze())
                )
                losses_dict["dice"] = dice_loss_weight * DiceLoss()(
                    fixed_seg.squeeze(), moving_seg.squeeze(), moved_seg.squeeze()
                )

            if use_keypoints:
                fixed_kps, moving_kps = (
                    torch.Tensor(data.fixed_keypoints).to(device),
                    torch.Tensor(data.moving_keypoints).to(device),
                )
                moved_kps = moving_kps + flow
                losses_dict["keypoints"] = TotalRegistrationLoss()(
                    (fixed_kps, moved_kps)
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
                    flownetc.state_dict(), checkpoint_dir / f"featnet_step{step}.pth"
                )

        torch.save(flownetc.state_dict(), checkpoint_dir / f"featnet_step{steps}.pth")

    @app.command()
    def eval(checkpoint: Path, data_json: Path, savedir: Path, device: str = "cuda"):
        savedir.mkdir(exist_ok=True)
        flownetc = FlowNetCorr()
        flownetc.load_state_dict(torch.load(checkpoint))
        flownetc = flownetc.to(device).eval()
        gen = data_generator(data_json, split="val")
        labels = load_labels(data_json)

        measurements = defaultdict(dict)

        for data in tqdm(gen):
            fixed_tio = tio.ScalarImage(data.fixed_image)
            moving_tio = tio.ScalarImage(data.moving_image)

            fixed = fixed_tio.data.unsqueeze(0).to(device)
            moving = moving_tio.data.unsqueeze(0).to(device)

            fixed_seg = tio.LabelMap(data.fixed_segmentation).data.unsqueeze(0)
            moving_seg = (
                tio.LabelMap(data.fixed_segmentation).data.unsqueeze(0).to(device)
            )

            flow = flownetc(moving, fixed)
            transformer = SpatialTransformer(fixed.shape[2:]).to(device)
            moved_seg = torch.round(transformer(moving_seg.float(), flow))

            disp_name = f"{data.moving_image.name}2{data.fixed_image.name}"
            moved_seg = torch.round(moved_seg).detach().cpu()

            measurements[disp_name]["dice"] = compute_dice(
                fixed_seg.numpy(),
                moving_seg.detach().cpu().numpy(),
                moved_seg.numpy(),
                labels=labels,
            )

        with open(savedir / "measurements.json", "w") as f:
            json.dump(measurements, f)

    app()
