import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.losses import FocalLoss
from common import get_identity_affine_grid


def swa_loop(H, W, D,
             net, grid0, optimizer,
             mind_fixed, mind_moving,
             lambda_weight,
             iterations,
             scheduler):

    for _ in iterations:

        focalloss = FocalLoss()
        focal_loss_weight = 10 if (_ < 20 or 60 < _ < 100) else 2.5
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

        floss = focalloss(patch_mov_sampled, mind_fixed) * focal_loss_weight

        total_loss = loss + reg_loss + floss

        total_loss.backward(retain_graph=True)

        optimizer.step()

        if _ > 70:
            scheduler.step()


def swa_optimization(
        disp,
        mind_fixed,
        mind_moving,
        lambda_weight,
        image_shape,
        norm,
):
    H, W, D = image_shape

    net = nn.Sequential(nn.Conv3d(3,1,(H, W, D)))
    net[0].weight.data[:] = disp / norm
    net.cuda()

    grid0 = get_identity_affine_grid(image_shape)
    optimizer = torch.optim.Adam(net.parameters(), lr=15)

    scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=0.5)

    swa_loop(H, W, D,
             net, grid0, optimizer,
             mind_fixed, mind_moving,
             lambda_weight,
             iterations=range(250),
             scheduler=scheduler)

    return net
