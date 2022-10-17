from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from monai.losses import FocalLoss
from torch.utils.tensorboard import SummaryWriter

from differentiable_metrics import DiceLoss
from common import get_identity_affine_grid

calc_dice = DiceLoss()

def tb_optimizer(
        writer: SummaryWriter,
        losses_dict: Dict[str, torch.Tensor],
        step: int,
) -> None:
    for loss_name, loss in losses_dict.items():
        writer.add_scalar(loss_name, loss, global_step=step)

def swa_loop(H, W, D,
             net, grid0, optimizer,
             mind_fixed, mind_moving,
             lambda_weight,
             iterations,
             schedulera,
             schedulerb,
             writer, img_name, fixed_keypoints, moving_keypoints,
             fixed_spacing, moving_spacing,
             fsegt, msegt):

    for _ in iterations:

        iteration_losses = {}

        # focalloss = FocalLoss()
        # focal_loss_weight = 10 if (_ < 20 or 60 < _ < 100) else 2.5
        # image_loss_weight = 3 if 100 < _ < 120 else 1

        optimizer.zero_grad()

        disp_sample = F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1)\
            .permute(0, 2, 3, 4, 1)

        reg_loss = (
                lambda_weight
                * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean()
                + lambda_weight
                * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean()
                + lambda_weight
                * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean())

        fitted_grid = disp_sample.permute(0, 4, 1, 2, 3).cuda()

        disp_hr = F.interpolate(
            fitted_grid * 2,
            size=(224, 192, 224),
            mode="trilinear",
            align_corners=False,
        )

        if fsegt is not None and msegt is not None:
            dice = calc_dice(fsegt, msegt, disp_hr)
            print("Torch " + str(dice.item()))
            iteration_losses["dice"] = dice.item()

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
        loss = sampled_cost.mean() # * image_loss_weight

        # floss = focalloss(patch_mov_sampled, mind_fixed) * focal_loss_weight

        total_loss = loss + reg_loss # + floss

        iteration_losses["loss"] = loss.item()
        iteration_losses["reg_loss"] = reg_loss.item()
        iteration_losses["total_loss"] = total_loss.item()

        tb_optimizer(writer=writer, losses_dict=iteration_losses, step=_)

        total_loss.backward(retain_graph=True)

        optimizer.step()

        if _ > 70:
            if 180<_<250:
                schedulerb.step()
            else:
                schedulera.step()

# def swa_loop_old(H, W, D,
#              net, grid0, optimizer,
#              mind_fixed, mind_moving,
#              lambda_weight,
#              iterations,
#              scheduler):
#
#     for _ in iterations:
#
#         focalloss = FocalLoss()
#         focal_loss_weight = 10 if (_ < 20 or 60 < _ < 100) else 2.5
#         # image_loss_weight = 3 if 100 < _ < 120 else 1
#
#         optimizer.zero_grad()
#
#         disp_sample = F.avg_pool3d(F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1)\
#             .permute(0, 2, 3, 4, 1)
#
#         reg_loss = (
#                 lambda_weight
#                 * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean()
#                 + lambda_weight
#                 * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean()
#                 + lambda_weight
#                 * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean())
#
#         scale = (torch.tensor([(H - 1) / 2, (W - 1) / 2, (D - 1) / 2, ]).cuda().unsqueeze(0))
#         grid_disp = (grid0.view(-1, 3).cuda().float()
#                      + ((disp_sample.view(-1, 3)) / scale).flip(1).float())
#
#         patch_mov_sampled = F.grid_sample(
#             mind_moving.float(),
#             grid_disp.view(1, H, W, D, 3).cuda(),
#             align_corners=False,
#             mode="bilinear",
#             padding_mode="border")
#
#         sampled_cost = (patch_mov_sampled - mind_fixed).pow(2).mean(1) * 12
#         loss = sampled_cost.mean() # * image_loss_weight
#
#         floss = focalloss(patch_mov_sampled, mind_fixed) * focal_loss_weight
#
#         total_loss = loss + reg_loss + floss
#
#         total_loss.backward(retain_graph=True)
#
#         optimizer.step()
#
#         if _ > 70:
#             scheduler.step()


def swa_optimization(
        disp,
        mind_fixed,
        mind_moving,
        lambda_weight,
        image_shape,
        norm,
        img_name, fkp, mkp, fs, checkpoint_dir,
        fsegt, msegt,
):
    H, W, D = image_shape

    checkpoint_dir = Path(checkpoint_dir / Path(img_name))
    checkpoint_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(checkpoint_dir))

    net = nn.Sequential(nn.Conv3d(3,1,(H, W, D)))
    # net[0].weight.data[:] = disp / norm
    net.cuda()

    grid0 = get_identity_affine_grid(image_shape)
    optimizer = torch.optim.Adam(net.parameters(), lr=15)

    schedulera = CosineAnnealingLR(optimizer, T_max=200, eta_min=0.1)
    schedulerb = LinearLR(optimizer, start_factor=0.999, end_factor=0.1, total_iters=70)

    swa_loop(H, W, D,
             net, grid0, optimizer,
             mind_fixed, mind_moving,
             lambda_weight,
             iterations=range(270),
             schedulera=schedulera, schedulerb=schedulerb,
             writer=writer, img_name=img_name, fixed_keypoints=fkp, moving_keypoints=mkp,
             fixed_spacing=fs, moving_spacing=fs,
             fsegt=fsegt, msegt=msegt,
             )

    return net
