from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import random
from typing import Generator, Optional, Tuple

import numpy as np
from scipy.ndimage import map_coordinates
import torch
from torch import nn
import torch.nn.functional as F


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
def get_identity_affine_grid(size: Tuple[int, ...], grid_sp: int) -> torch.Tensor:
    """
    Computes an identity grid for a specific size.

    Parameters
    ----------
    size: Tuple[int,...]
    """

    H, W, D = size

    grid0 = F.affine_grid(
        torch.eye(3, 4).unsqueeze(0).cuda(),
        [1, 1, H // grid_sp, W // grid_sp, D // grid_sp],
        align_corners=False,
    )
    return grid0


def apply_displacement_field(disp_field: np.ndarray, image: np.ndarray) -> np.ndarray:
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
    moved_image = map_coordinates(image, id_grid + disp_field, order=1)
    if has_channel:
        moved_image = moved_image[None, ...]
    return moved_image


def correlate(
    mind_fix: torch.Tensor,
    mind_mov: torch.Tensor,
    disp_hw: int,
    grid_sp: int,
    image_shape: Tuple[int, int, int],
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
    H, W, D = image_shape
    torch.cuda.synchronize()
    C_mind = mind_fix.shape[1]
    with torch.no_grad():
        mind_unfold = F.unfold(
            F.pad(
                mind_mov, (disp_hw, disp_hw, disp_hw, disp_hw, disp_hw, disp_hw)
            ).squeeze(0),
            disp_hw * 2 + 1,
        )
        mind_unfold = mind_unfold.view(
            C_mind, -1, (disp_hw * 2 + 1) ** 2, W // grid_sp, D // grid_sp
        )

    ssd = torch.zeros(
        (disp_hw * 2 + 1) ** 3,
        H // grid_sp,
        W // grid_sp,
        D // grid_sp,
        dtype=mind_fix.dtype,
        device=mind_fix.device,
    )  # .cuda().half()
    ssd_argmin = torch.zeros(H // grid_sp, W // grid_sp, D // grid_sp).long()
    with torch.no_grad():
        for i in range(disp_hw * 2 + 1):
            mind_sum = (
                (mind_fix.permute(1, 2, 0, 3, 4) - mind_unfold[:, i : i + H // grid_sp])
                .abs()
                .sum(0, keepdim=True)
            )

            ssd[i :: (disp_hw * 2 + 1)] = F.avg_pool3d(
                mind_sum.transpose(2, 1), 3, stride=1, padding=1
            ).squeeze(1)
        ssd = (
            ssd.view(
                disp_hw * 2 + 1,
                disp_hw * 2 + 1,
                disp_hw * 2 + 1,
                H // grid_sp,
                W // grid_sp,
                D // grid_sp,
            )
            .transpose(1, 0)
            .reshape((disp_hw * 2 + 1) ** 3, H // grid_sp, W // grid_sp, D // grid_sp)
        )
        ssd_argmin = torch.argmin(ssd, 0)  #
        # ssd = F.softmax(-ssd*1000,0)
    torch.cuda.synchronize()

    return ssd, ssd_argmin


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

    coeffs = torch.tensor([0.003, 0.01, 0.03, 0.1, 0.3, 1])
    for j in range(6):
        ssd_coupled_argmin = torch.zeros_like(ssd_argmin)
        with torch.no_grad():
            for i in range(H // grid_sp):

                coupled = ssd[:, i, :, :] + coeffs[j] * (
                    disp_mesh_t - disp_soft[:, :, i].view(3, 1, -1)
                ).pow(2).sum(0).view(-1, W // grid_sp, D // grid_sp)
                ssd_coupled_argmin[i] = torch.argmin(coupled, 0)
            # print(coupled.shape)

        disp_soft = F.avg_pool3d(
            disp_mesh_t.view(3, -1)[:, ssd_coupled_argmin.view(-1)].reshape(
                1, 3, H // grid_sp, W // grid_sp, D // grid_sp
            ),
            3,
            padding=1,
            stride=1,
        )

    return disp_soft


# enforce inverse consistency of forward and backward transform
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

def MINDSEG(imseg, shape, weight):
    """
    Not entirely sure what is going on here, tbh
    """

    mindssc = (
        10
        * (
            F.one_hot(imseg.cuda().view(1, *shape).long())
            .float()
            .permute(0, 4, 1, 2, 3)
            .contiguous()
            * weight.view(1, -1, 1, 1, 1).cuda()
        ).half()
    )

    return mindssc


def MINDSSC(img: torch.Tensor, radius: int=2, dilation: int=2):
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
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = (x > y).view(-1) & (dist == 2).view(-1)
    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
    mshift1.view(-1)[
        torch.arange(12) * 27
        + idx_shift1[:, 0] * 9
        + idx_shift1[:, 1] * 3
        + idx_shift1[:, 2]
    ] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
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
    mind /= mind_var
    mind = torch.exp(-mind)
    # permute to have same ordering as C++ code
    mind = mind[:, torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
    return mind

def compute_loss(
    disp_lr: torch.Tensor,
    mind_fixed: torch.Tensor,
    mind_moving: torch.Tensor,
    lambda_weight: float,
    grid_sp: int,
    image_shape: Tuple[int, int, int],
    ) -> torch.Tensor:
    """
    """
    ...
    disp_sample = disp_lr
    H, W, D = image_shape

    grid0 = get_identity_affine_grid(image_shape, grid_sp=grid_sp)

    reg_loss = (
        lambda_weight
        * ((disp_sample[0, :, 1:, :] - disp_sample[0, :, :-1, :]) ** 2).mean()
        + lambda_weight
        * ((disp_sample[0, 1:, :, :] - disp_sample[0, :-1, :, :]) ** 2).mean()
        + lambda_weight
        * ((disp_sample[0, :, :, 1:] - disp_sample[0, :, :, :-1]) ** 2).mean()
    )

    scale = (
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
    grid_disp = (
        grid0.view(-1, 3).cuda().float()
        + ((disp_sample.view(-1, 3)) / scale).flip(1).float()
    )

    patch_mov_sampled = F.grid_sample(
        mind_moving.float(),
        grid_disp.view(1, H // grid_sp, W // grid_sp, D // grid_sp, 3).cuda(),
        align_corners=False,
        mode="bilinear",
    )  # ,padding_mode='border')
    sampled_cost = (patch_mov_sampled - mind_fixed).pow(2).mean(1) * 12

    loss = sampled_cost.mean() + reg_loss
    return loss


def adam_optimization(
    disp_lr: torch.Tensor,
    mind_fixed: torch.Tensor,
    mind_moving: torch.Tensor,
    lambda_weight: float,
    grid_sp: int,
    image_shape: Tuple[int, int, int],
    iterations: int,
) -> nn.Module:
    """
    Instance-based optimization

    Parameters
    ----------
    disp_lr: torch.Tensor,
    mind_fixed: torch.Tensor,
    mind_moving: torch.Tensor,
    lambda_weight: float,
    grid_sp: int,
    image_shape: Tuple[int, int, int],
    iterations: int,

    Returns
    -------
    nn.Module
    """

    H, W, D = image_shape
    # create optimisable displacement grid
    net = nn.Sequential(
        nn.Conv3d(3, 1, (H // grid_sp, W // grid_sp, D // grid_sp), bias=False)
    )
    net[0].weight.data[:] = disp_lr / grid_sp
    net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1)
    grid0 = F.affine_grid(
        torch.eye(3, 4).unsqueeze(0).cuda(),
        [1, 1, H // grid_sp, W // grid_sp, D // grid_sp],
        align_corners=False,
    )

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
        grid_disp = (
            grid0.view(-1, 3).cuda().float()
            + ((disp_sample.view(-1, 3)) / scale).flip(1).float()
        )

        patch_mov_sampled = F.grid_sample(
            mind_moving.float(),
            grid_disp.view(1, H // grid_sp, W // grid_sp, D // grid_sp, 3).cuda(),
            align_corners=False,
            mode="bilinear",
        )  # ,padding_mode='border')
        sampled_cost = (patch_mov_sampled - mind_fixed).pow(2).mean(1) * 12

        loss = sampled_cost.mean()
        (loss + reg_loss).backward()
        optimizer.step()

    return net


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


def random_never_ending_generator(data_json: Path) -> Generator[Data, None, None]:
    """
    Generator that 1) never ends and 2) yields samples from the dataset in random order

    Parameters
    ----------
    data_json: Path
        JSON file containing data information

    """

    with open(data_json, "r") as f:
        data = json.load(f)["data"]

    segs = "fixed_segmentation" in data[0]
    kps = "fixed_keypoints" in data[0]

    while True:
        random.shuffle(data)
        for v in data:
            yield Data(
                fixed_image=Path(v["fixed_image"]),
                moving_image=Path(v["moving_image"]),
                fixed_segmentation=Path(v["fixed_segmentation"]) if segs else None,
                moving_segmentation=Path(v["moving_segmentation"]) if segs else None,
                fixed_keypoints=Path(v["fixed_keypoints"]) if kps else None,
                moving_keypoints=Path(v["moving_keypoints"]) if kps else None,
            )

def data_generator(data_json: Path) -> Generator[Data, None, None]:
    """
    Generator function.

    Parameters
    ----------
    data_json: JSON file containing data information
    """
    with open(data_json, "r") as f:
        data = json.load(f)["data"]

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
