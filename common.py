from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import json
from pathlib import Path
import random
from typing import Dict, Generator, List, Optional, Tuple
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from scipy.ndimage import map_coordinates
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard.writer import SummaryWriter


class DisplacementFormat(str, Enum):
    Nifti = "nifti"
    Numpy = "numpy"


def torch2skimage_disp(disp_field: torch.Tensor) -> np.ndarray:
    x1 = disp_field[0, 0, :, :, :].cpu().float().data.numpy()
    y1 = disp_field[0, 1, :, :, :].cpu().float().data.numpy()
    z1 = disp_field[0, 2, :, :, :].cpu().float().data.numpy()

    displacement = np.stack([x1, y1, z1], -1)
    return displacement


def tb_log(
    writer: SummaryWriter,
    losses_dict: Dict[str, float],
    step: int,
    moving_fixed_moved: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    for loss_name, loss in losses_dict.items():
        writer.add_scalar(loss_name, loss, global_step=step)

    moving, fixed, moved = moving_fixed_moved
    slice_index = moving.shape[2] // 2
    triplet = [moving.squeeze(), fixed.squeeze(), moved.squeeze()]
    writer.add_images(
        "(moving,fixed,moved)",
        img_tensor=torch.stack(triplet)[:, :, slice_index, ...].unsqueeze(1),
        global_step=step,
        dataformats="nchw",
    )


@lru_cache(maxsize=None)
def identity_grid(size: Tuple[int, ...]) -> np.ndarray:
    """
    Computes an identity grid for a specific size.

    Parameters
    ----------
    size: Tuple[int,...]
    """
    vectors = [np.arange(0, s) for s in size]
    grids = np.meshgrid(*vectors, indexing="ij")
    grid = np.stack(grids, axis=0)
    return grid


@lru_cache(maxsize=None)
def identity_grid_torch(size: Tuple[int, ...], device: str="cuda", stack_dim: int=0) -> torch.Tensor:
    """
    Computes an identity grid for torch
    """
    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors, indexing="ij")
    grid = torch.stack(grids, dim=stack_dim)
    grid = torch.unsqueeze(grid, 0).float().to(device)

    return grid


@lru_cache(maxsize=None)
def displacement_permutations_grid(displacement: int) -> torch.Tensor:
    disp_mesh_t = (
        F.affine_grid(
            displacement * torch.eye(3, 4).unsqueeze(0),
            [1, 1, displacement * 2 + 1, displacement * 2 + 1, displacement * 2 + 1],
            align_corners=True,
        )
        .permute(0, 4, 1, 2, 3)
        .reshape(3, -1, 1)
    )

    return disp_mesh_t


@lru_cache(maxsize=None)
def get_identity_affine_grid(size: Tuple[int, ...]) -> torch.Tensor:
    """
    Computes an identity grid for a specific size.

    Parameters
    ----------
    size: Tuple[int,...]
    """

    H, W, D = size

    grid0 = F.affine_grid(
        torch.eye(3, 4).unsqueeze(0).cuda(), [1, 1, H, W, D], align_corners=False,
    )
    return grid0


def get_labels(
    fixed_seg: torch.Tensor, moving_seg: torch.Tensor, include_zero: bool = False
) -> list:
    fixed_labels = (torch.unique(fixed_seg.long())).tolist()
    moving_labels = (torch.unique(moving_seg.long())).tolist()
    label_list = list((np.unique(set(fixed_labels + moving_labels)))[0])
    if not include_zero:
        label_list.remove(0)
    return label_list


def apply_displacement_field(
    disp_field: np.ndarray, image: np.ndarray, order: int = 1
) -> np.ndarray:
    """
    Applies displacement field to the image

    Parameters
    ----------
    disp_field: np.ndarray
        Must be compatible with skimage's map_coordinates function
    image: np.ndarray
        
    Returns
    -------
    moved_image: np.ndarray
    """
    has_channel = len(image.shape) == 4
    if has_channel:
        image = image[0, ...]
    size = image.shape
    assert len(size) == 3

    id_grid = identity_grid(size)
    moved_image = map_coordinates(image, id_grid + disp_field, order=order)
    if has_channel:
        moved_image = moved_image[None, ...]
    return moved_image

def unravel_indices(
    indices: torch.LongTensor, shape: Tuple[int, ...]
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []
    ndim = len(shape)

    for i, dim in enumerate(reversed(shape)):
        coord.append(indices % dim)
        indices = torch.floor(indices / dim)

    coord = torch.stack(coord[::-1], dim=-1)

    return coord  # type: ignore


def compute_interpolation_weights(zyx_warped):
    zyx_warped = zyx_warped.view(-1,3)
    z_warped = zyx_warped[:,0]
    y_warped = zyx_warped[:,1]
    x_warped = zyx_warped[:,2]

    z_f = torch.floor(z_warped)
    z_c = z_f + 1

    y_f = torch.floor(y_warped)
    y_c = y_f + 1

    x_f = torch.floor(x_warped)
    x_c = x_f + 1

    w00 = (y_c - y_warped) * (x_c - x_warped)
    w01 = (y_warped - y_f) * (x_c - x_warped)
    w02 = (y_c - y_warped) * (x_warped - x_f)
    w03 = (y_warped - y_f) * (x_warped - x_f)

    w0 = (z_c - z_warped) * w00
    w1 = (z_warped-z_f) * w00
    w2 = (z_c - z_warped) * w01
    w3 = (z_warped-z_f) * w01
    w4 = (z_c - z_warped) * w02
    w5 = (z_warped-z_f) * w02
    w6 = (z_c - z_warped) * w03
    w7 = (z_warped-z_f) * w03

    weights = [ w0, w1, w2, w3, w4, w5, w6, w7, ]
    indices = [
        torch.stack([z_f, y_f, x_f]),
        torch.stack([z_c, y_f, x_f]),
        torch.stack([z_f, y_c, x_f]),
        torch.stack([z_c, y_c, x_f]),
        torch.stack([z_f, y_f, x_c]),
        torch.stack([z_c, y_f, x_c]),
        torch.stack([z_f, y_c, x_c]),
        torch.stack([z_c, y_c, x_c]),
    ]
    weights = torch.stack(weights, dim=0)
    indices = torch.stack(indices, dim=1)

    return weights, indices

def correlate(
    mind_fix: torch.Tensor,
    mind_mov: torch.Tensor,
    search_radius: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes dense discretised displacements to compute SSD cost volume with box-filter

    Parameters
    ----------
    mind_mov: torch.Tensor
    mind_fix: torch.Tensor
    disp_hw: int
    grid_sp: int
    image_shape: Tuple[int, int, int]

    Returns
    -------
    ssd: torch.Tensor
        Sum of square displacements
    ssd_argmin: torch.Tensor
        Sum of square displacements min
    """
    H, W, D = mind_fix.shape[-3:]
    torch.cuda.synchronize()
    C_mind = mind_fix.shape[1]
    with torch.no_grad():
        mind_unfold = F.unfold(
            F.pad(
                mind_mov, (search_radius, search_radius, search_radius, search_radius, search_radius, search_radius)
            ).squeeze(0),
            search_radius * 2 + 1,
        )
        mind_unfold = mind_unfold.view(
            C_mind, -1, (search_radius * 2 + 1) ** 2, W, D
        )

    ssd = torch.zeros( 
            (search_radius * 2 + 1) ** 3,
            H, W, D,
            dtype=mind_fix.dtype, device=mind_fix.device,)  # .cuda().half()
    ssd_argmin = torch.zeros(H , W , D).long()
    with torch.no_grad():
        for i in range(search_radius * 2 + 1):
            mind_sum = (
                (mind_fix.permute(1, 2, 0, 3, 4) - mind_unfold[:, i : i + H])
                .abs()
                .sum(0, keepdim=True)
            )

            ssd[i :: (search_radius * 2 + 1)] = F.avg_pool3d(
                mind_sum.transpose(2, 1), 3, stride=1, padding=1
            ).squeeze(1)
        ssd = (
            ssd.view(
                search_radius * 2 + 1,
                search_radius * 2 + 1,
                search_radius * 2 + 1,
                H, W, D,
            )
            .transpose(1, 0)
            .reshape((search_radius * 2 + 1) ** 3, H , W , D )
        )
        ssd_argmin = torch.argmin(ssd, 0)  #
        # ssd = F.softmax(-ssd*1000,0)
    torch.cuda.synchronize()

    return ssd, ssd_argmin


def gumbel_softmax(logits: torch.Tensor, temperature: float = 0.8) -> torch.Tensor:
    """
    Implements straight through gumbel softmax

    Parameters
    ----------
    logits: torch.Tensor
        Log likelihood for each class with shape [*, n_class]
    temperature: float
        The temperature controls how much smoothing there is, between 0 and 1
        Default=.8

    Returns
    -------
    one_hot: torch.Tensor
        One-hot tensor that can be used to sample discrete class tensor
    """
    # FIXME: what exactly is the role of temperature here
    # FIXME: is the output always one-hot

    def gumbel_softmax_sample(logits, temperature):
        y = logits + sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def sample_gumbel(shape, eps=1e-20):
        U = torch.rand(shape).cuda()
        return -Variable(torch.log(-torch.log(U + eps) + eps))

    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y



def coupled_convex_sparse(
    distances: torch.Tensor,
    displacements: torch.Tensor,
    image_shape: Tuple[int, int, int],
    K=10,
) -> torch.Tensor:
    """
    Solve two coupled convex optimisation problems for efficient global regularisation

    Parameters
    ----------
    ssd: torch.Tensor
    ssd_argmin: torch.Tensor
    disp_mesh_t: torch.Tensor
        All possible permutations
    grid_sp: int
    image_shape: Tuple[int, int, int]

    Returns
    -------
    disp_soft: torch.Tensor
    """
    H, W, D = image_shape

    disp_soft = F.avg_pool3d(displacements[0,...].float(), 3, padding=1, stride=1,)

    # reg_coeffs = torch.tensor([0.003, 0.01, 0.03, 0.1, 0.3, 1])
    reg_coeffs = torch.tensor([0.003, 0.01, 0.03, 0.1, 0.3, 1])
    for coeff in reg_coeffs:
        ssd_coupled_argmin = torch.zeros(image_shape).to(displacements.device)
        for i in range(H):
            with torch.no_grad():
                coupled = distances[:,i,:,:] + coeff*(displacements[:,:,i,...]-disp_soft[:,i,...].unsqueeze(0)).pow(2).sum(1).view(-1, W, D)
                ssd_coupled_argmin[i] = torch.argmin(coupled, 0)

        disp_hard = (
                F.one_hot(ssd_coupled_argmin.view(-1).long(), num_classes=K)
                .permute(1,0).unsqueeze(1)*displacements.view(K,3,-1)).sum(0).reshape(3, *image_shape).float()
        disp_soft = F.avg_pool3d(disp_hard, 3, padding=1, stride=1)

    return disp_soft



def coupled_convex(
    ssd: torch.Tensor,
    ssd_argmin: torch.Tensor,
    disp_mesh_t: torch.Tensor,
    grid_sp: int,
    image_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Solve two coupled convex optimisation problems for efficient global regularisation

    Parameters
    ----------
    ssd: torch.Tensor
    ssd_argmin: torch.Tensor
    disp_mesh_t: torch.Tensor
        All possible permutations
    grid_sp: int
    image_shape: Tuple[int, int, int]

    Returns
    -------
    disp_soft: torch.Tensor
    """
    H, W, D = image_shape
    disp_soft = F.avg_pool3d(
        disp_mesh_t.view(3, -1)[:, ssd_argmin.view(-1)].reshape(
            1, 3, H // grid_sp, W // grid_sp, D // grid_sp
        ),
        3,
        padding=1,
        stride=1,
    )

    reg_coeffs = torch.tensor([0.003, 0.01, 0.03, 0.1, 0.3, 1])
    for coeff in reg_coeffs:
        ssd_coupled_argmin = torch.zeros_like(ssd_argmin)
        with torch.no_grad():
            for i in range(H // grid_sp):
                coupled = ssd[:, i, :, :] + coeff * (
                    disp_mesh_t - disp_soft[:, :, i].view(3, 1, -1)
                ).pow(2).sum(0).view(-1, W // grid_sp, D // grid_sp)
                ssd_coupled_argmin[i] = torch.argmin(coupled, 0)

        disp_soft = F.avg_pool3d(
            disp_mesh_t.view(3, -1)[:, ssd_coupled_argmin.view(-1)].reshape(
                1, 3, H // grid_sp, W // grid_sp, D // grid_sp
            ),
            3,
            padding=1,
            stride=1,
        )

    return disp_soft


def inverse_consistency(
    disp_field1s: torch.Tensor, disp_field2s: torch.Tensor, iter: int = 20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ensures inverse consistency

    Parameters
    ----------
    disp_field1s: torch.Tensor
    disp_field2s: torch.Tensor
    iter: int
        Default: 20

    Returns:
    --------
    disp_field1_ic: torch.Tensor
    disp_field2_ic: torch.Tensor
    """
    # factor = 1
    _, _, H, W, D = disp_field1s.size()
    # make inverse consistent
    with torch.no_grad():
        disp_field1i = disp_field1s.clone()
        disp_field2i = disp_field2s.clone()

        identity = (
            F.affine_grid(torch.eye(3, 4).unsqueeze(0), [1, 1, H, W, D])
            .permute(0, 4, 1, 2, 3)
            .to(disp_field1s.device)
            .to(disp_field1s.dtype)
        )
        for _ in range(iter):
            disp_field1s = disp_field1i.clone()
            disp_field2s = disp_field2i.clone()

            disp_field1i = 0.5 * (
                disp_field1s
                - F.grid_sample(
                    disp_field2s, (identity + disp_field1s).permute(0, 2, 3, 4, 1)
                )
            )
            disp_field2i = 0.5 * (
                disp_field2s
                - F.grid_sample(
                    disp_field1s, (identity + disp_field2s).permute(0, 2, 3, 4, 1)
                )
            )

    return disp_field1i, disp_field2i


def pdist_squared(x):
    xx = (x ** 2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist, 0.0)  # , np.inf)
    return dist


def MINDSEG(
    imseg: torch.Tensor,
    shape: Tuple[int, ...],
    norm_weight: torch.Tensor,
    feature_weight: float = 10.0,
):
    """
    Compute one-hot features using segmentations

    Parameters
    ----------
    imseg: torch.Tensor
    shape: Tuple[int,...]
        shape of the segmentation tensor
    norm_weight: torch.Tensor
    feature_weight: float
        A hyperparameter to control the feature space loss.

    Returns
    -------
    segfeats: torch.Tensor
    """

    mindssc = (
        feature_weight
        * (
            F.one_hot(
                imseg.cuda().view(1, *shape).long(), num_classes=norm_weight.shape[0]
            )
            .float()
            .permute(0, 4, 1, 2, 3)
            .contiguous()
            * norm_weight.view(1, -1, 1, 1, 1).cuda()
        ).half()
    )
    # import pdb; pdb.set_trace()
    return mindssc


def MINDSSC(
    img: torch.Tensor, radius: int = 2, dilation: int = 2, device: str = "cuda"
):
    """
    Computes local structural features.

    See http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf

    Parameters
    ----------
    img: torch.Tensor
    """
    # kernel size
    kernel_size = radius * 2 + 1
    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.tensor(
        [[0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2], [2, 1, 1], [1, 2, 1]]
    ).long()
    # squared distances
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij")
    mask = (x > y).view(-1) & (dist == 2).view(-1)
    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).to(device)
    mshift1.view(-1)[
        torch.arange(12) * 27
        + idx_shift1[:, 0] * 9
        + idx_shift1[:, 1] * 3
        + idx_shift1[:, 2]
    ] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).to(device)
    mshift2.view(-1)[
        torch.arange(12) * 27
        + idx_shift2[:, 0] * 9
        + idx_shift2[:, 1] * 3
        + idx_shift2[:, 2]
    ] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)
    # compute patch-ssd
    ssd = F.avg_pool3d(
        rpad2(
            (
                F.conv3d(rpad1(img), mshift1, dilation=dilation)
                - F.conv3d(rpad1(img), mshift2, dilation=dilation)
            )
            ** 2
        ),
        kernel_size,
        stride=1,
    )
    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.min(
        torch.max(mind_var, mind_var.mean() * 0.001), mind_var.mean() * 1000
    )
    mind /= mind_var + 1e-8
    mind = torch.exp(-mind)
    # permute to have same ordering as C++ code
    mind = mind[:, torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
    return mind


class UnrolledConv(nn.Module):
    def __init__(
        self, n_cascades: int, image_shape: Tuple[int, int, int], grid_sp: int
    ):
        super().__init__()
        self.n_cascades = n_cascades
        self.convs = nn.ModuleList(
            [nn.Conv3d(3, 3, 3, padding="same") for _ in range(n_cascades)]
        )

        # FIXME: remove these tensors
        self.shape: Tuple[int, ...] = image_shape
        H, W, D = image_shape
        self.grid0 = F.affine_grid(
            torch.eye(3, 4).unsqueeze(0).cuda(),
            [1, 1, H // grid_sp, W // grid_sp, D // grid_sp],
            align_corners=False,
        )
        self.grid_sp: int = grid_sp

        self.scale = (
            torch.tensor(
                [
                    (H // grid_sp - 1) / 2,
                    (W // grid_sp - 1) / 2,
                    (D // grid_sp - 1) / 2,
                ]
            )
            .cuda()
            .unsqueeze(0)
        )

    def forward(self, feat_mov: torch.Tensor, disp_sample: torch.Tensor):
        H, W, D = self.shape
        for i in range(self.n_cascades):
            disp_sample = self.convs[i](disp_sample)
            grid_disp = (
                self.grid0.view(-1, 3).cuda().float()
                + ((disp_sample.view(-1, 3)) / self.scale).flip(1).float()
            )
            feat_mov = F.grid_sample(
                feat_mov.float(),
                grid_disp.view(
                    1, H // self.grid_sp, W // self.grid_sp, D // self.grid_sp, 3
                ).cuda(),
                align_corners=False,
                mode="bilinear",
            )

        return disp_sample, feat_mov


def swa_optimization(
    disp: torch.Tensor,
    mind_fixed: torch.Tensor,
    mind_moving: torch.Tensor,
    lambda_weight: float,
    image_shape: Tuple[int, int, int],
    iterations: int,
    norm: int = 1,
) -> nn.Module:
    """
    Instance-based optimization

    Parameters
    ----------
    disp_lr: torch.Tensor,
    mind_fixed: torch.Tensor,
    mind_moving: torch.Tensor,
    lambda_weight: float,
    image_shape: Tuple[int, int, int],
    iterations: int,

    Returns
    -------
    nn.Module
    """

    H, W, D = image_shape
    # create optimisable displacement grid
    net = nn.Sequential(nn.Conv3d(3, 1, (H, W, D), bias=False))
    net[0].weight.data[:] = disp / norm
    net.cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)

    swa_net = AveragedModel(net)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    swa_start = iterations // 2
    swa_scheduler = SWALR(optimizer, swa_lr=1)

    grid0 = get_identity_affine_grid(image_shape)

    for i in range(iterations):
        optimizer.zero_grad()

        disp_sample = F.avg_pool3d(
            F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1,
        ).permute(0, 2, 3, 4, 1)
        reg_loss = (
            lambda_weight
            * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean()
            + lambda_weight
            * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean()
            + lambda_weight
            * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()
        )

        scale = (
            torch.tensor([(H - 1) / 2, (W - 1) / 2, (D - 1) / 2,]).cuda().unsqueeze(0)
        )
        grid_disp = (
            grid0.view(-1, 3).cuda().float()
            + ((disp_sample.view(-1, 3)) / scale).flip(1).float()
        )

        patch_mov_sampled = F.grid_sample(
            mind_moving.float(),
            grid_disp.view(1, H, W, D, 3).cuda(),
            align_corners=False,
            mode="bilinear",
        )  # ,padding_mode='border')
        sampled_cost = (patch_mov_sampled - mind_fixed).pow(2).mean(1) * 12
        loss = sampled_cost.mean()
        (loss + reg_loss).backward(retain_graph=True)
        optimizer.step()

        if i > swa_start:
            swa_net.update_parameters(net)
            swa_scheduler.step()
        else:
            scheduler.step()

    return net

def adam_optimization_teo(
    disp: torch.Tensor,
    mind_fixed: torch.Tensor,
    mind_moving: torch.Tensor,
    lambda_weight: float,
    image_shape: Tuple[int, int, int],
    iterations: int,
    norm: int = 1,
) -> nn.Module:
    """
    Instance-based optimization

    Parameters
    ----------
    disp_lr: torch.Tensor,
    mind_fixed: torch.Tensor,
    mind_moving: torch.Tensor,
    lambda_weight: float,
    image_shape: Tuple[int, int, int],
    iterations: int,

    Returns
    -------
    nn.Module
    """

    H, W, D = image_shape
    # create optimisable displacement grid
    net = nn.Sequential(nn.Conv3d(3, 1, (H, W, D), bias=False))
    net[0].weight.data[:] = disp / norm
    net.cuda()
    grid0 = get_identity_affine_grid(image_shape)
    for iterations, lr in zip([70, 60, 60, 60], [1, 5, 2, 1]):
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        for _ in range(iterations):
            optimizer.zero_grad()

            disp_sample = F.avg_pool3d(
                F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1,
            ).permute(0, 2, 3, 4, 1)
            reg_loss = (
                lambda_weight
                * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean()
                + lambda_weight
                * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean()
                + lambda_weight
                * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()
            )

            scale = (
                torch.tensor([(H - 1) / 2, (W - 1) / 2, (D - 1) / 2,]).cuda().unsqueeze(0)
            )
            grid_disp = (
                grid0.view(-1, 3).cuda().float()
                + ((disp_sample.view(-1, 3)) / scale).flip(1).float()
            )

            patch_mov_sampled = F.grid_sample(
                mind_moving.float(),
                grid_disp.view(1, H, W, D, 3).cuda(),
                align_corners=False,
                mode="bilinear",
            )  # ,padding_mode='border')
            sampled_cost = (patch_mov_sampled - mind_fixed).pow(2).mean(1) * 12

            loss = sampled_cost.mean()
            (loss + reg_loss).backward(retain_graph=True)
            optimizer.step()

    return net




def adam_optimization(
    disp: torch.Tensor,
    mind_fixed: torch.Tensor,
    mind_moving: torch.Tensor,
    lambda_weight: float,
    image_shape: Tuple[int, int, int],
    iterations: int,
    norm: int = 1,
) -> nn.Module:
    """
    Instance-based optimization

    Parameters
    ----------
    disp_lr: torch.Tensor,
    mind_fixed: torch.Tensor,
    mind_moving: torch.Tensor,
    lambda_weight: float,
    image_shape: Tuple[int, int, int],
    iterations: int,

    Returns
    -------
    nn.Module
    """

    H, W, D = image_shape
    # create optimisable displacement grid
    net = nn.Sequential(nn.Conv3d(3, 1, (H, W, D), bias=False))
    net[0].weight.data[:] = disp / norm
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1)
    grid0 = get_identity_affine_grid(image_shape)

    for _ in range(iterations):
        optimizer.zero_grad()

        disp_sample = F.avg_pool3d(
            F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1,
        ).permute(0, 2, 3, 4, 1)
        reg_loss = (
            lambda_weight
            * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean()
            + lambda_weight
            * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean()
            + lambda_weight
            * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()
        )

        scale = (
            torch.tensor([(H - 1) / 2, (W - 1) / 2, (D - 1) / 2,]).cuda().unsqueeze(0)
        )
        grid_disp = (
            grid0.view(-1, 3).cuda().float()
            + ((disp_sample.view(-1, 3)) / scale).flip(1).float()
        )

        patch_mov_sampled = F.grid_sample(
            mind_moving.float(),
            grid_disp.view(1, H, W, D, 3).cuda(),
            align_corners=False,
            mode="bilinear",
        )  # ,padding_mode='border')
        sampled_cost = (patch_mov_sampled - mind_fixed).pow(2).mean(1) * 12

        loss = sampled_cost.mean()
        (loss + reg_loss).backward(retain_graph=True)
        optimizer.step()

    return net


class TrainType(str, Enum):
    Paired = "paired"
    Disregard = "disregard"
    Unpaired = "unpaired"
    UnpairedSet = "unpairedset"


@dataclass
class Data:
    """
    Generic Data class for storing data information
    """

    fixed_image: Path
    moving_image: Path
    fixed_segmentation: Optional[Path]
    moving_segmentation: Optional[Path]
    fixed_keypoints: Optional[Path]
    moving_keypoints: Optional[Path]


def load_keypoints(keypoints_path: Path) -> torch.Tensor:
    with open(keypoints_path, "r") as f:
        arr = np.array(
            [[float(j) for j in i.strip().split(",")] for i in f.readlines()]
        )
    # if not (torch.floor(torch.from_numpy(arr)) == torch.ceil(torch.from_numpy(arr))).all():
    #     raise ValueError("Keypoints must be integers")
    return torch.from_numpy(arr)


def load_keypoints_np(keypoints_path: Path) -> np.ndarray:
    with open(keypoints_path, "r") as f:
        arr = np.array(
            [[float(j) for j in i.strip().split(",")] for i in f.readlines()]
        )

    return arr


def load_labels(data_json: Path) -> List[int]:
    with open(data_json, "r") as f:
        labels = json.load(f)["labels"]

    labels = [int(i) for i in labels]
    return labels


def random_unpaired_split_never_ending_generator(
    data_json: Path, *, seed: Optional[int] = None
) -> Generator[Data, None, None]:
    """
    This generator is used when you have a set of fixed images and a set of moving
    images and you want to randomly sample fixed and moving images from their respective sets.

    Parameters
    ----------
    data_json: Path
        JSON file containing data information
    split: str
        One of train or val
    seed: Optional[int]
        Default=None

    """
    fixed_split = "train_fixed"
    moving_split = "train_moving"
    if seed:
        random.seed(seed)

    with open(data_json, "r") as f:
        data_fixed = json.load(f)[fixed_split]

    with open(data_json, "r") as f:
        data_moving = json.load(f)[moving_split]

    while True:
        random.shuffle(data_fixed)
        random.shuffle(data_moving)
        for fixed, moving in zip(data_fixed, data_moving):

            fixed_image = fixed["image"]
            moving_image = moving["image"]

            segmentation = "label" in fixed and "label" in moving
            keypoints = "keypoints" in fixed and "keypoints" in moving

            yield Data(
                fixed_image=fixed_image,
                moving_image=moving_image,
                fixed_segmentation=fixed["label"] if segmentation else None,
                moving_segmentation=moving["label"] if segmentation else None,
                fixed_keypoints=fixed["keypoints"] if keypoints else None,
                moving_keypoints=moving["keypoints"] if keypoints else None,
            )


def random_unpaired_never_ending_generator(
    data_json: Path, *, split: str, seed: Optional[int] = None
) -> Generator[Data, None, None]:
    """
    Generator that randomly generates pairs from an unpaird dataset.

    Parameters
    ----------
    data_json: Path
        JSON file containing data information
    split: str
        One of train or val
    seed: Optional[int]
        Default=None

    """
    if seed:
        random.seed(seed)

    with open(data_json, "r") as f:
        data = json.load(f)[split]

    while True:
        random.shuffle(data)
        for i in range(0, len(data), 2):
            fixed = data[i]
            moving = data[i + 1]

            fixed_image = fixed["image"]
            moving_image = moving["image"]

            segmentation = "label" in fixed and "label" in moving
            keypoints = "keypoints" in fixed and "keypoints" in moving

            yield Data(
                fixed_image=fixed_image,
                moving_image=moving_image,
                fixed_segmentation=fixed["label"] if segmentation else None,
                moving_segmentation=moving["label"] if segmentation else None,
                fixed_keypoints=fixed["keypoints"] if keypoints else None,
                moving_keypoints=moving["keypoints"] if keypoints else None,
            )


def randomized_pair_never_ending_generator(
    data_json: Path, *, split: str, seed: Optional[int] = None
) -> Generator[Data, None, None]:
    """
    Generator that completely disregards pairs defined in the JSON.

    Parameters
    ----------
    data_json: Path
        JSON file containing data information
    split: str
        One of train or val
    seed: Optional[int]
        Default=None

    """
    if seed:
        random.seed(seed)

    with open(data_json, "r") as f:
        data = json.load(f)[split]

    images: List[Dict[str, Optional[Path]]] = [
        {
            "image_path": Path(d[f"{i}_image"]),
            "seg_path": Path(d[f"{i}_segmentation"])
            if f"{i}_segmentation" in d
            else None,
            "kps_path": Path(d[f"{i}_keypoints"]) if f"{i}_keypoints" in d else None,
        }
        for d in data
        for i in ("fixed", "moving")
    ]

    while True:
        fixed = random.choice(images)
        moving = random.choice(images)

        fixed_image = fixed["image_path"]
        moving_image = moving["image_path"]
        assert fixed_image is not None and moving_image is not None

        yield Data(
            fixed_image=fixed_image,
            moving_image=moving_image,
            fixed_segmentation=fixed["seg_path"],
            moving_segmentation=moving["seg_path"],
            fixed_keypoints=fixed["kps_path"],
            moving_keypoints=moving["kps_path"],
        )


def random_never_ending_generator(
    data_json: Path,
    *,
    split: str,
    random_switch: bool = False,
    seed: Optional[int] = None,
) -> Generator[Data, None, None]:
    """
    Generator that 1) never ends and 2) yields samples from the dataset in random order

    Parameters
    ----------
    data_json: Path
        JSON file containing data information
    split: str
        One of train or val
    seed: Optional[int]
        Default=None
    """
    if seed:
        random.seed(seed)

    with open(data_json, "r") as f:
        data = json.load(f)[split]

    segs = "fixed_segmentation" in data[0]
    kps = "fixed_keypoints" in data[0]

    f, m = "fixed", "moving"
    if random_switch and random.randint(0, 10) % 2 == 0:
        f, m = m, f

    while True:
        random.shuffle(data)
        for v in data:
            yield Data(
                fixed_image=Path(v[f"{f}_image"]),
                moving_image=Path(v[f"{m}_image"]),
                fixed_segmentation=Path(v[f"{f}_segmentation"]) if segs else None,
                moving_segmentation=Path(v[f"{m}_segmentation"]) if segs else None,
                fixed_keypoints=Path(v[f"{f}_keypoints"]) if kps else None,
                moving_keypoints=Path(v[f"{m}_keypoints"]) if kps else None,
            )


def data_generator(data_json: Path, *, split: str) -> Generator[Data, None, None]:
    """
    Generator function.

    Parameters
    ----------
    data_json: JSON file containing data information
    split: str
        One of train or val
    """
    with open(data_json, "r") as f:
        data = json.load(f)[split]

    # FIXME: make this cleaner
    segs = "fixed_segmentation" in data[0]
    kps = "fixed_keypoints" in data[0]

    for v in data:
        yield Data(
            fixed_image=Path(v["fixed_image"]),
            moving_image=Path(v["moving_image"]),
            fixed_segmentation=Path(v["fixed_segmentation"]) if segs else None,
            moving_segmentation=Path(v["moving_segmentation"]) if segs else None,
            fixed_keypoints=Path(v["fixed_keypoints"]) if kps else None,
            moving_keypoints=Path(v["moving_keypoints"]) if kps else None,
        )


def concat_flow(
    flow1: torch.Tensor, flow2: torch.Tensor, mode: str="bilinear"
):
    grid = identity_grid_torch(flow1.shape[-3:]).to(flow1.device)
    locs1 = grid + flow1
    new_locs = grid + flow2

    shape = flow2.shape[2:]

    # need to normalize grid values to [-1, 1] for resampler
    for i in range(len(shape)):
        new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

    if len(shape) == 2:
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
    elif len(shape) == 3:
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

    new_grid = torch.stack([
        F.grid_sample(locs1[:,i,...].unsqueeze(1), new_locs,mode="bilinear").squeeze(1)
        for i in range(3)], dim=1)
    new_flow = new_grid - grid
    return new_flow

def warp_image(
    displacement_field: torch.Tensor, image: torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    grid = identity_grid_torch(image.shape[-3:], device=image.device)
    new_locs = grid + displacement_field

    shape = displacement_field.shape[2:]

    # need to normalize grid values to [-1, 1] for resampler
    for i in range(len(shape)):
        new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

    if len(shape) == 2:
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
    elif len(shape) == 3:
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

    resampled = F.grid_sample(image, new_locs, align_corners=True, mode=mode)
    return resampled

