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
from networks import Unet3D, default_unet_features, ResizeTransform

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
        redir_feats: int=32,
    ):
        super().__init__()
        self.correlation_patch_size = correlation_patch_size
        self.feature_extractor = FeatureExtractor(  # (*fex_args, **fex_kwargs)
            infeats=infeats,
            feature_sizes=feature_extractor_feature_sizes,
            kernel_sizes=feature_extractor_kernel_sizes,
            strides=feature_extractor_strides,
        )
        self.conv_redir =_conv_bn_lrelu(
            infeats=feature_extractor_feature_sizes[-1],
            outfeats=redir_feats,
            kernel_size=1,
            stride=1,
        )
        # TODO: this is not strictly like flownet because we just pop in a unet
        # I just got lazy 😅
        corr_out_feat = (1+(correlation_patch_size*2))**3 + redir_feats
        self.unet = Unet3D(infeats=corr_out_feat)
        self.flow = nn.Conv3d(default_unet_features()[-1][-1], 3, kernel_size=3, padding=1)

        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape)) 
        resize_factor = 1/np.prod(feature_extractor_strides).item()
        self.flow_resize = ResizeTransform(resize_factor, ndims=3)

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
        return flow

        
def _conv_bn_lrelu(infeats: int, outfeats: int, kernel_size: int, stride: int):
    padding = 'same' if stride == 1 else 0
    return nn.Sequential(
        nn.Conv3d(
            in_channels=infeats,
            out_channels=outfeats,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        ),
        nn.BatchNorm3d(outfeats),
        nn.LeakyReLU()
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

    from tqdm import tqdm, trange
    from torch.utils.tensorboard.writer import SummaryWriter
    import torchio as tio
    import typer

    from common import data_generator, random_never_ending_generator
    from differentiable_metrics import MutualInformationLoss, DiceLoss, Grad
    from metrics import compute_dice
    from networks import SpatialTransformer

    app = typer.Typer()

    @app.command()
    def test_forward_pass(data_json: Path):
        gen = data_generator(data_json)
        data = next(gen)

        fixed_tio = tio.ScalarImage(data.fixed_image)
        moving_tio = tio.ScalarImage(data.moving_image)

        fixed = fixed_tio.data.unsqueeze(0)
        moving = moving_tio.data.unsqueeze(0)

        fnet = FlowNetCorr()
        flow = fnet(moving, fixed)
        SpatialTransformer(fixed.shape)(moving.unsqueeze(0).unsqueeze(0), flow)

    @app.command()
    def train(data_json: Path, 
              checkpoint_dir: Path,
              labels: List[int],
              steps: int=1000, 
              lr: float=3e-4,
              device: str="cuda",
              mi_loss_weight: float=1, 
              dice_loss_weight: float=2., 
              reg_loss_weight: float=.1,
              log_freq: int=5,
              save_freq: int=100,
        ):
        gen = random_never_ending_generator(data_json)
        checkpoint_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir=checkpoint_dir)
        fnet = FlowNetCorr().to(device)
        opt = torch.optim.Adam(fnet.parameters(), lr=lr)
        loss_names = ["mi", "dice", "reg"]
        loss_fns = [MutualInformationLoss(), DiceLoss(labels), Grad()]
        loss_weights = [mi_loss_weight, dice_loss_weight, reg_loss_weight]
        for step in trange(steps):
            data = next(gen)

            fixed_tio = tio.ScalarImage(data.fixed_image)
            moving_tio = tio.ScalarImage(data.moving_image)

            fixed = fixed_tio.data.unsqueeze(0).to(device)
            moving = moving_tio.data.unsqueeze(0).to(device)

            fixed_seg = tio.LabelMap(data.fixed_segmentation).data.unsqueeze(0).to(device)
            moving_seg = tio.LabelMap(data.fixed_segmentation).data.unsqueeze(0).to(device)

            flow = fnet(moving, fixed)
            transformer = SpatialTransformer(fixed.shape[2:]).to(device)
            moved = transformer(moving, flow)
            moved_seg = torch.round(transformer(moving_seg.float(), flow))

            loss_inputs = [(moved, fixed), (fixed_seg.squeeze(), moving_seg.squeeze(), moved_seg.squeeze()), [flow]]
            losses: List[torch.Tensor] = [w*fn(*inp) for fn, w, inp in zip(loss_fns, loss_weights, loss_inputs)]

            opt.zero_grad()
            loss = sum(losses)
            loss.backward()
            opt.step()

            if step % log_freq == 0:
                for loss_name, loss in zip(loss_names, losses):
                    writer.add_scalar(loss_name, loss, global_step=step)

                slice_index = moving.shape[2]//2
                triplet = [moving.squeeze(), fixed.squeeze(), moved.squeeze()]
                writer.add_images(
                    "(moving,fixed,moved)",
                    img_tensor=torch.stack(triplet)[:, slice_index, ...].unsqueeze(1),
                    global_step=step,
                    dataformats="nchw",
                )

            if step % save_freq == 0:
                torch.save(fnet.state_dict(), checkpoint_dir/f"featnet_step{step}.pth")

        torch.save(fnet.state_dict(), checkpoint_dir/f"featnet_step{steps}.pth")


    @app.command()
    def eval(checkpoint: Path, data_json: Path, savedir: Path, labels: List[int]=[1], device: str="cuda"):
        savedir.mkdir(exist_ok=True)
        fnet = FlowNetCorr()
        fnet.load_state_dict(torch.load(checkpoint))
        fnet = fnet.to(device).eval()
        gen = data_generator(data_json)

        measurements = defaultdict(dict)

        for data in tqdm(gen):
            fixed_tio = tio.ScalarImage(data.fixed_image)
            moving_tio = tio.ScalarImage(data.moving_image)

            fixed = fixed_tio.data.unsqueeze(0).to(device)
            moving = moving_tio.data.unsqueeze(0).to(device)

            fixed_seg = tio.LabelMap(data.fixed_segmentation).data.unsqueeze(0)
            moving_seg = tio.LabelMap(data.fixed_segmentation).data.unsqueeze(0).to(device)

            flow = fnet(moving, fixed)
            transformer = SpatialTransformer(fixed.shape[2:]).to(device)
            moved_seg = torch.round(transformer(moving_seg.float(), flow))

            disp_name = f"{data.moving_image.name}2{data.fixed_image.name}"
            moved_seg = torch.round(moved_seg).detach().cpu()

            measurements[disp_name]["dice"] = compute_dice(
                fixed_seg.numpy(), moving_seg.detach().cpu().numpy(), moved_seg.numpy(), labels=labels
            )

        with open(savedir / "measurements.json", "w") as f:
            json.dump(measurements, f)

    app()
