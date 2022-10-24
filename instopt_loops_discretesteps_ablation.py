import numpy as np
import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.tensorboard.writer import SummaryWriter
from pathlib import Path
from typing import Dict

from monai.losses import FocalLoss
from monai.losses.dice import DiceLoss as monaiDice
from torch.nn import BCELoss

from common import apply_displacement_field, warp_image
from differentiable_metrics import TotalRegistrationLoss, DiceLoss
from metrics import compute_dice


def tb_optimizer(
        writer: SummaryWriter,
        losses_dict: Dict[str, torch.Tensor],
        step: int,
) -> None:
    for loss_name, loss in losses_dict.items():
        writer.add_scalar(loss_name, loss, global_step=step)


trl = TotalRegistrationLoss()
calc_dice = DiceLoss()
bce_loss = BCELoss(reduction="mean")
monaidice = monaiDice(include_background=False)

def opt_loop(H, W, D,
             net, grid0, optimizer,
             mind_fixed, mind_moving,
             lambda_weight,
             iterations,
             writer, img_name, fixed_keypoints, moving_keypoints,
             fixed_spacing, moving_spacing,
            fsegt, msegt, labels, fullres=False):

    for _ in iterations:
        iteration_losses = {}

        optimizer.zero_grad()

        disp_sample = F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1)\
            .permute(0, 2, 3, 4, 1)
        # disp_sample.retain_grad()

        reg_loss = (
                lambda_weight
                * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean()
                + lambda_weight
                * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean()
                + lambda_weight
                * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean())

        # if fsegt is not None and msegt is not None:
        #     dice = calc_dice(fsegt, msegt, disp_sample.permute(0, 4, 1, 2, 3))
        #     # dice.retain_grad()
        #     iteration_losses["dice"] = dice.item()
        #     print(dice.item())
        #
        #     # wsegt = warp_image(disp_sample.permute(0, 4, 1, 2, 3), msegt, mode="nearest")
        #     # bce = bce_loss(fsegt, wsegt)
        #     # bce.retain_grad()
        #     # iteration_losses["bce"] = bce.item()
        #     # print(bce.item())

        if (int(img_name) < 101) and (fixed_keypoints is not None):
            fitted_grid = disp_sample.permute(0, 4, 1, 2, 3)

            disp_hr = F.interpolate(
                fitted_grid * 2,
                size=(H * 2, W * 2, D * 2),
                mode="trilinear",
                align_corners=False,
            )
            disp_hr.retain_grad()

            disp_sample_full = einops.rearrange(disp_hr, "b c d h w -> (b c) d h w")

            tre = trl(
                fixed_keypoints,
                moving_keypoints,
                disp_sample_full.unsqueeze(0),
                fixed_spacing,
                moving_spacing
            )
            print(tre.item())
            iteration_losses["tre"] = tre.item()

        scale = (torch.tensor([(H - 1) / 2, (W - 1) / 2, (D - 1) / 2, ]).cuda().unsqueeze(0))
        grid_disp = (grid0.view(-1, 3).cuda().float()
                     + ((disp_sample.view(-1, 3)) / scale).flip(1).float())

        patch_mov_sampled = F.grid_sample(
            mind_moving.float(),
            grid_disp.view(1, H, W, D, 3).cuda(),
            align_corners=False,
            mode="bilinear",
            padding_mode="border")

        sampled_cost = (patch_mov_sampled - mind_fixed).pow(2).mean(1) * 12
        loss = sampled_cost.mean()

        total_loss = loss + reg_loss # + dice  #+ floss
        # total_loss.retain_grad()

        iteration_losses["loss"] = loss.item()
        iteration_losses["reg_loss"] = reg_loss.item()
        iteration_losses["total_loss"] = total_loss.item()
        # iteration_losses["floss"] = floss.item()

        tb_optimizer(writer=writer, losses_dict=iteration_losses, step=_)
        total_loss.backward(retain_graph=True)
        # bce.backward()

        optimizer.step()


def lr_step_optimization(
        disp,
        mind_fixed,
        mind_moving,
        lambda_weight,
        image_shape,
        checkpoint_dir,
        img_name,
        norm=None, fkp=None, mkp=None, fs=None, grid0=None,
        fsegt=None, msegt=None, labels=None,
):
    H, W, D = image_shape

    checkpoint_dir = Path(checkpoint_dir / Path(img_name))
    checkpoint_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(checkpoint_dir))

    if fsegt is not None:
        labels = fsegt.unique()

        fsegs, msegs = [], []

        for i in labels:
            if i == 0:
            # if not (i.item() ==1):
                # if i.item() in [0, 16, 18, 32, 34]:
                continue
            fseginter, mseginter = (fsegt == i).float(), (msegt == i).float()
            fsegs.append(fseginter)
            msegs.append(mseginter)
        assert len(fsegs) != 0, "No labels found!"

        fseg = torch.cat(fsegs, dim=1)
        mseg = torch.cat(msegs, dim=1)

    net = nn.Sequential(nn.Conv3d(3,1,(H, W, D)))
    net[0].weight.data[:] = disp / norm # comment out if not setting initial weights of network to anything
    net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=1)
    opt_loop(H, W, D,
             net, grid0, optimizer,
             mind_fixed, mind_moving,
             lambda_weight,
             iterations=range(0,70),
             writer=writer, img_name=img_name,
             fixed_keypoints=fkp,
             moving_keypoints=mkp,
             fixed_spacing=fs, moving_spacing=fs,
             fsegt=fseg if fsegt is not None else None,
             msegt=mseg if fsegt is not None else None, labels=None)
    optimizer = torch.optim.Adam(net.parameters(), lr=5)
    opt_loop(H, W, D,
             net, grid0, optimizer,
             mind_fixed, mind_moving,
             lambda_weight,
             iterations=range(70,130),
             writer=writer, img_name=img_name,
             fixed_keypoints=fkp,
             moving_keypoints=mkp,
             fixed_spacing=fs, moving_spacing=fs,
             fsegt=fseg if fsegt is not None else None,
             msegt=mseg if fsegt is not None else None, labels=None)
    optimizer = torch.optim.Adam(net.parameters(), lr=2)
    opt_loop(H, W, D,
             net, grid0, optimizer,
             mind_fixed, mind_moving,
             lambda_weight,
             iterations=range(130,190),
             writer=writer, img_name=img_name,
             fixed_keypoints=fkp,
             moving_keypoints=mkp,
             fixed_spacing=fs, moving_spacing=fs,
             fsegt=fseg if fsegt is not None else None,
             msegt=mseg if fsegt is not None else None, labels=None)
    optimizer = torch.optim.Adam(net.parameters(), lr=1)
    opt_loop(H, W, D,
             net, grid0, optimizer,
             mind_fixed, mind_moving,
             lambda_weight,
             iterations=range(190,250),
             writer=writer, img_name=img_name,
             fixed_keypoints=fkp,
             moving_keypoints=mkp,
             fixed_spacing=fs, moving_spacing=fs,
             fsegt=fseg if fsegt is not None else None,
             msegt=mseg if fsegt is not None else None, labels=None)

    return net
