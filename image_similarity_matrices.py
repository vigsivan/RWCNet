import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from common import MINDSSC

__all__ = ["mi", "ncc", "mse", "mind_mse"]

def mind_mse(y_true: torch.Tensor, y_pred: torch.Tensor):
    return mse(MINDSSC(y_true), MINDSSC(y_pred))


def mse(y_true: torch.Tensor, y_pred: torch.Tensor):
    return (y_true - y_pred) ** 2


def mi(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Args:
        pred: the shape should be B[NDHW].
        target: the shape should be same as the pred shape.
    Raises:
        ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
    """
    smooth_nr, smooth_dr = 1e-7, 1e-7
    if target.shape != pred.shape:
        raise ValueError(f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})")
    wa, pa = _parzen_windowing_gaussian(pred)
    wb, pb = _parzen_windowing_gaussian(target)

    pab = torch.bmm(wa.permute(0, 2, 1), wb.to(wa)).div(wa.shape[1])  # (batch, num_bins, num_bins)
    papb = torch.bmm(pa.permute(0, 2, 1), pb.to(pa))  # (batch, num_bins, num_bins)
    mi = torch.sum(
        pab * torch.log((pab + smooth_nr) / (papb + smooth_dr) + smooth_dr), dim=(1, 2)
    )
    return mi

def ncc(y_true, y_pred):
    Ii = y_true
    Ji = y_pred

    # get dimension of volume
    # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
    ndims = len(list(Ii.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # set window size
    win = [9] * ndims

    # compute filters
    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0] / 2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1, 1)
        padding = (pad_no, pad_no)
    else:
        stride = (1, 1, 1)
        padding = (pad_no, pad_no, pad_no)

    # get convolution function
    conv_fn = getattr(F, 'conv%dd' % ndims)

    # compute CC squares
    I2 = Ii * Ii
    J2 = Ji * Ji
    IJ = Ii * Ji

    I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
    J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
    I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
    J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
    IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = cross * cross / (I_var * J_var + 1e-5)

    return cc

def _parzen_windowing_gaussian(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parzen windowing with gaussian kernel (adapted from DeepReg implementation)
    Note: the input is expected to range between 0 and 1
    Args:
        img: the shape should be B[NDHW].
    """
    num_bins=23
    sigma_ratio = .5

    bin_centers = torch.linspace(0.0, 1.0, num_bins)  # (num_bins,)
    sigma = torch.mean(bin_centers[1:] - bin_centers[:-1]) * sigma_ratio
    preterm = 1 / (2 * sigma**2)


    img = torch.clamp(img, 0, 1)
    img = img.reshape(img.shape[0], -1, 1)  # (batch, num_sample, 1)
    weight = torch.exp(
        -preterm.to(img) * (img - bin_centers.to(img)) ** 2
    )  # (batch, num_sample, num_bin)
    weight = weight / torch.sum(weight, dim=-1, keepdim=True)  # (batch, num_sample, num_bin)
    probability = torch.mean(weight, dim=-2, keepdim=True)  # (batch, 1, num_bin)
    return weight, probability

