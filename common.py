from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import json
from pathlib import Path
import random
from typing import Dict, Generator, List, Optional, Tuple
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

import einops
import numpy as np
from scipy.ndimage import map_coordinates
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard.writer import SummaryWriter


from knn import knn_faiss_raw


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
        img_tensor=torch.stack(triplet)[:, slice_index, ...].unsqueeze(1),
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


def correlate_grad(
    feat_fix: torch.Tensor,
    feat_mov: torch.Tensor,
    disp_hw: int,
    grid_sp: int,
    image_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    A differentiable version of the correlation function that computes the sum of
    square distances between patches in the moving image and the fixed image.
    """
    H, W, D = image_shape
    torch.cuda.synchronize()
    C_feat = feat_fix.shape[1]
    with torch.no_grad():
        feat_unfold = F.unfold(
            F.pad(
                feat_mov, (disp_hw, disp_hw, disp_hw, disp_hw, disp_hw, disp_hw)
            ).squeeze(0),
            disp_hw * 2 + 1,
        )
        feat_unfold = feat_unfold.view(
            C_feat, -1, (disp_hw * 2 + 1) ** 2, W // grid_sp, D // grid_sp
        )

    ssd_list = []
    for i in range(disp_hw * 2 + 1):
        feat_sum = (
            (feat_fix.permute(1, 2, 0, 3, 4) - feat_unfold[:, i : i + H // grid_sp])
            .abs()
            .sum(0, keepdim=True)
        )

        ssd_list.append(
            F.avg_pool3d(feat_sum.transpose(2, 1), 3, stride=1, padding=1).squeeze()
        )

    ssd_list2 = []
    for item_num in range(ssd_list[0].shape[0]):
        for bucket_num in range(len(ssd_list)):
            ssd_list2.append(ssd_list[bucket_num][item_num, ...])

    ssd = torch.stack(ssd_list2)

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

    return ssd


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


def correlate_sparse_unrolled(
        feat_fix: torch.Tensor, feat_mov: torch.Tensor, K: int = 10, radius: int=16
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert feat_fix.shape == feat_mov.shape, "Fixed and moving feature shapes must be the same!"

    kc, kh, kw = [radius]*3
    dc, dh, dw = [radius]*3

    patches_fix = feat_fix.unfold(2, kc, dc).unfold(3, kh, dh).unfold(4, kw, dw)
    patches_mov = feat_mov.unfold(2, kc, dc).unfold(3, kh, dh).unfold(4, kw, dw)
    nchan = feat_fix.shape[1]

    pfix = patches_fix.contiguous().view(1, nchan,-1, kc * kh * kw)
    pmov = patches_mov.contiguous().view(1, nchan,-1, kc * kh * kw)
    npatches = pfix.shape[-2]

    patch_shape = (kc, kh, kw)

    grid = identity_grid_torch(patch_shape, feat_fix.device, stack_dim=-1)
    grid = grid.unsqueeze(1).float()

    dist_patched = torch.zeros((1, K, npatches, kc, kh, kw))
    disp_patched = torch.zeros((1, K, npatches, kc, kh, kw, 3))
    for i in range(npatches):
        dist, ind = knn_faiss_raw(pfix[...,i,:].float(), pmov[...,i,:].float(), K)

        dist = einops.rearrange(dist, 'b K (h w d) -> b K h w d', h=kh, w=kw)
        ind3d = unravel_indices(ind, patch_shape)
        ind3d = einops.rearrange(ind3d, 'b K (h w d) D -> b K h w d D', h=kh, w=kw)
        displacement = ind3d-grid
        dist_patched[:,:,i,...] = dist
        disp_patched[:,:,i,...] = displacement

    dist_patched_re = dist_patched.view(1, K, *patches_fix.shape[2:])
    disp_patched_re = disp_patched.view(1, K, *patches_fix.shape[2:], 3)

    dist_patched_re = dist_patched_re.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous()
    disp_patched_re = disp_patched_re.permute(0, 1, 2, 5, 3, 6, 4, 7, 8).contiguous()

    feature_distances = dist_patched_re.view(1, K, *feat_fix.shape[-3:])
    displacements = disp_patched_re.view(1, K, *feat_fix.shape[-3:], 3)

    displacements = einops.rearrange(displacements, 'b K h d w N -> (b K) N h d w')
    return feature_distances.squeeze(), displacements.long()


def correlate_sparse(
        feat_fix: torch.Tensor, feat_mov: torch.Tensor, K: int = 10, num_splits=4, radius: int=10
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert feat_fix.shape == feat_mov.shape, "Fixed and moving feature shapes must be the same!"
    assert feat_fix.shape[-3]%num_splits == 0, \
            f"Number of splits must be a factor of the first spatial dim {feat_fix.shape[-3]}"
    arrs_fix = torch.tensor_split(feat_fix, num_splits, -3)
    arrs_mov = torch.tensor_split(feat_mov, num_splits, -3)
    split_size = feat_fix.shape[-3]//num_splits

    size = (split_size, *feat_fix.shape[-2:])
    grid = identity_grid_torch(size, feat_fix.device, stack_dim=-1)
    grid = grid.unsqueeze(1).float()

    distances, displacements = [], []
    for arr_fix, arr_mov in zip(arrs_fix, arrs_mov):
        arr_fix = einops.rearrange(arr_fix, 'b c h d w -> b c (h d w)') 
        arr_mov = einops.rearrange(arr_mov, 'b c h d w -> b c (h d w)') 
        dist, ind = knn_faiss_raw(arr_fix.float(), arr_mov.float(), K)
        
        dist = einops.rearrange(dist, 'b K (h d w) -> b K h d w', h=split_size, d=feat_fix.shape[-2])
        ind3d = unravel_indices(ind, size)
        ind3d = einops.rearrange(ind3d, 'b K (h d w) D-> b K h d w D', h=split_size, d=feat_fix.shape[-2])
        displacement = ind3d-grid
        
        distances.append(dist)
        displacements.append(displacement)

    dist = torch.concat(distances, dim=-3).squeeze()
    displacement = torch.concat(displacements, dim=-4).squeeze()

    displacement = einops.rearrange(displacement, 'K h d w N -> K N h d w')
    return dist, displacement.long()

def correlate_sparse_with_splitting(
    feat_fix: torch.Tensor, feat_mov: torch.Tensor, K: int = 10, patch_size: int = 8, patches: int=4
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert feat_fix.shape[-2]%patches == 0, "Num patches must be a factor of the first spatial dimension"

    arrs_mov = torch.tensor_split(feat_mov, patches, -2)

    dists, inds = [], []
    for arr_fix, arr_mov in zip(arrs_fix, arrs_mov):
        arr_fix = einops.rearrange(arr_fix, 'b c h D -> b c (h D)') 
        arr_mov = einops.rearrange(arr_mov, 'b c h D -> b c (h D)') 
        dist, ind = knn_faiss_raw(arr_fix.float(), arr_mov.float(), K)
        dist = einops.rearrange(dist, 'b K (h D) -> b K h D', h=patches)
        ind = einops.rearrange(ind, 'b K (h D) -> b K h D', h=patches)

        dists.append(dist)
        inds.append(ind)

    # TODO: how do we get the right indexes here?

    breakpoint()

    assert feat_fix.shape == feat_mov.shape, "Input features must be the same size"
    assert (
        feat_fix.shape[0] == 1 and len(feat_fix.shape) == 5
    ), "Input features must be 5d bcdhw"

    patches_fix = einops.rearrange(
        feat_fix.unfold(dimension=-3, size=patch_size, step=patch_size)
        .unfold(dimension=-3, size=patch_size, step=patch_size)
        .unfold(dimension=-3, size=patch_size, step=patch_size),
        "b f nx ny nz px py pz -> b (nx ny nz) f (px py pz)",
    )
    patches_mov = einops.rearrange(
        feat_mov.unfold(dimension=-3, size=patch_size, step=patch_size)
        .unfold(dimension=-3, size=patch_size, step=patch_size)
        .unfold(dimension=-3, size=patch_size, step=patch_size),
        "b f nx ny nz px py pz -> b (nx ny nz) f (px py pz)",
    )

    breakpoint()

    npatches = patches_fix.shape[1]
    dist = torch.zeros((1, npatches, K, patches_fix.shape[-1])).to(patches_fix.device)
    patch_position = torch.zeros((1, npatches, K, patches_fix.shape[-1])).to(
        patches_fix.device
    )
    for i in range(npatches):
        di, pi = knn_faiss_raw(
            patches_fix[:, i, ...].float(), patches_mov[:, i, ...].float(), K
        )
        dist[:, i, ...] = di
        patch_position[:, i, ...] = pi

    size = feat_fix.shape[-3:]
    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors, indexing="ij")
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.float()

    patch_indices = (
        grid.unfold(dimension=-3, size=patch_size, step=patch_size)
        .unfold(dimension=-3, size=patch_size, step=patch_size)
        .unfold(dimension=-3, size=patch_size, step=patch_size)
    )

    patch_indices = einops.rearrange(
        patch_indices, "b f nx ny nz px py pz -> b f (nx ny nz) (px py pz)"
    ).to(feat_fix.device)
    patch_position = einops.rearrange(patch_position, "b p K s -> b p (K s)")

    global_positions = []
    for i in range(patch_position.shape[-2]):
        pos = torch.index_select(
            patch_indices[..., i, :], dim=-1, index=patch_position[0, i, :].long()
        )
        global_positions.append(pos)

    global_position = torch.stack(global_positions, dim=1)
    global_position = einops.rearrange(
        global_position,
        "b p d (K s) -> b d p K s",
        K=K,
        # s1=feat_fix.shape[-3],
        # s2=feat_fix.shape[-2],
    )
    disps = (global_position - patch_indices.unsqueeze(-2))
    F.fold(disps[0,...,0,:], output_size=32, kernel_size=8)

    breakpoint()
    dist = einops.rearrange(dist, 'b p K s -> b (p K s)')
    disps = einops.rearrange(disps, 'b d p K s -> (b d)(p K s)')


    return dist, disps


def correlate_sparse_bs(
    feat_fix: torch.Tensor, feat_mov: torch.Tensor, K: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert feat_fix.shape == feat_mov.shape, "Input features must be the same size"
    assert (
        feat_fix.shape[0] == 1 and len(feat_fix.shape) == 5
    ), "Input features must be 5d bcdhw"
    dists, indexes = [], []
    for ax in range(-3, 0):
        feat_fix = torch.swapaxes(feat_fix, -3, ax)
        feat_mov = torch.swapaxes(feat_mov, -3, ax)
        indices_agg = []
        disps_agg = []
        for i in range(feat_fix.shape[-3]):
            fix = einops.rearrange(feat_fix[..., i, :, :], "b c x y -> b c (x y)")
            mov = einops.rearrange(feat_mov[..., i, :, :], "b c x y -> b c (x y)")
            disps, index = knn_faiss_raw(fix.float(), mov.float(), K)
            disps_agg.append(disps)
            indices_agg.append(index)

        dist = torch.stack(disps_agg, dim=-2)
        index = torch.stack(indices_agg, dim=-2)

        dist = einops.rearrange(dist, "b K h (w d) -> b K h w d", d=feat_fix.shape[-1])
        index = einops.rearrange(
            index, "b K h (w d) -> b K h w d", d=feat_fix.shape[-1]
        )

        dist = torch.swapaxes(dist, -3, ax)
        index = torch.swapaxes(index, -3, ax)

        feat_fix = torch.swapaxes(feat_fix, -3, ax)
        feat_mov = torch.swapaxes(feat_mov, -3, ax)

        dist = einops.rearrange(dist, "b K h w d -> b K (h w d)")
        index = einops.rearrange(index, "b K h w d -> b K (h w d)")
        dists.append(dist)
        indexes.append(index)

    K = 3 * K
    dist = torch.concat(dists, 1)
    index = torch.concat(indexes, 1)
    indices = unravel_indices(
        einops.rearrange(index, "b K i -> b (K i)").squeeze().long(),  # type: ignore
        feat_fix.shape[-3:],
    )

    indices = einops.rearrange(indices, "(b K i) d -> b K i d", b=1, K=K)

    ogs = unravel_indices(
        torch.LongTensor([i for i in range(np.prod(feat_mov.shape[-3:]))]),
        feat_mov.shape[-3:],
    ).to(feat_mov.device)
    disps = indices - ogs
    dist = dist.view(1, K, *feat_fix.shape[-3:])
    disps = disps.view(1, K, *feat_fix.shape[-3:], 3).permute(0, -1, 1, 2, 3, 4)

    return dist, disps


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
    breakpoint()
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
    swa_scheduler = SWALR(optimizer, swa_lr=5)

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


def warp_image(
    displacement_field: torch.Tensor, image: torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    grid = identity_grid_torch(image.shape[-3:]).to(image.device)
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


if __name__ == "__main__":
    import typer
    import nibabel as nib
    from metrics import compute_total_registration_error

    app = typer.Typer()
    add_bc_dim = lambda x: einops.repeat(x, "d h w -> b c d h w", b=1, c=1).float()
    get_spacing = lambda x: np.sqrt(np.sum(x.affine[:3, :3] * x.affine[:3, :3], axis=0))

    @app.command()
    def visualize_correlate_sparse(data_json):
        data = next(data_generator(data_json, split="train"))
        fixed_nib = nib.load(data.fixed_image)
        moving_nib = nib.load(data.moving_image)

        fixed = add_bc_dim(torch.from_numpy(fixed_nib.get_fdata())).cuda().squeeze()
        moving = add_bc_dim(torch.from_numpy(moving_nib.get_fdata())).cuda().squeeze()

        fixed = (fixed - fixed.min())/(fixed.max() - fixed.min())
        moving = (moving - moving.min())/(moving.max() - moving.min())

        mind_fix = MINDSSC(
            fixed.unsqueeze(0).unsqueeze(0).cuda(), 1, 2
        ).half()
        mind_mov = MINDSSC(
            moving.unsqueeze(0).unsqueeze(0).cuda(), 1, 2
        ).half()

        mind_fix.requires_grad = True
        mind_mov.requires_grad = True

        costs, displacements = correlate_sparse_unrolled(mind_fix, mind_mov, K=10, num_splits=16)

        fixed_keypoints = np.loadtxt(data.fixed_keypoints, delimiter=",")
        moving_keypoints = np.loadtxt(data.moving_keypoints, delimiter=",")

        fixed_spacing = get_spacing(fixed_nib)
        moving_spacing = get_spacing(moving_nib)

        d0 = displacements[0].detach().cpu().numpy()
        d0 = einops.rearrange(d0, 'n h w d -> h w d n')
        print(compute_total_registration_error(fixed_keypoints, moving_keypoints, d0, fixed_spacing, moving_spacing))


    app()
