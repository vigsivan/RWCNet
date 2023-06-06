"""
Implements differentiable loss functions for training
"""

import warnings
import math
from collections import defaultdict
from functools import partial
from typing import List

import einops
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
# from monai.losses.dice import DiceLoss
from monai.losses.image_dissimilarity import GlobalMutualInformationLoss as MutualInformationLoss
from common import MINDSSC, warp_image

__all__ = ["DiceLoss", "MutualInformationLoss", "TotalRegistrationLoss", "Grad", "NCC", "MSE", "MINDLoss"]

class DiceLoss(nn.Module):
    """
    Computes the dice loss
    Note: This function does not include the background (assumed to be 0)
    """
    def __init__(self):
        super().__init__()

    def forward(self, fixed_seg: torch.Tensor, moving_seg: torch.Tensor, flow: torch.Tensor):
        labels = fixed_seg.unique()

        fsegs, msegs = [], []

        for i in labels:
            # NOTE: we assume background is label 0
            if i.item() == 0:
                continue
            fseg, mseg = (fixed_seg == i).float(), (moving_seg == i).float()
            fsegs.append(fseg)
            msegs.append(mseg)

        if len(fsegs) == 0:
            warnings.warn("No labels found!", RuntimeWarning)
            return None

        fseg = torch.cat(fsegs, dim=1)
        mseg = torch.cat(msegs, dim=1)

        wseg = warp_image(flow, mseg, "nearest")
        dice = _compute_dice_coefficient(fseg, wseg)
        return 1-dice

def _compute_dice_coefficient(mask_gt: torch.Tensor, mask_pred: torch.Tensor) -> torch.Tensor:
    """Computes soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`.

  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN.
  """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return torch.Tensor(0, device=mask_gt.device)
    volume_intersect = (mask_gt * mask_pred).sum()
    return 2 * volume_intersect / volume_sum


class TotalRegistrationLoss(nn.Module):
    """
    Computes the Total Registration Loss

    Which is basically the distance between the landmarks in the transformed space.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        fixed_landmarks: torch.LongTensor,
        moving_landmarks: torch.LongTensor,
        displacement_field: torch.Tensor,
        fixed_spacing: torch.Tensor,
        moving_spacing: torch.Tensor,
    ) -> torch.Tensor:
        #FIXME: this is hacky and likely slower than it should be, because PT doesn't have grid_sample
        #FIXME: figure out why the spacing was passed in
        #NOTE: I only do linear interpolation, so this could be made more accurate.

        # TODO: verify this implementation

        fixed_landmarks = fixed_landmarks.to(displacement_field.device)
        moving_landmarks = moving_landmarks.to(displacement_field.device)
        moving_spacing = moving_spacing.to(displacement_field.device)

        assert fixed_landmarks.shape == moving_landmarks.shape
        fixed_landmarks_tmp = fixed_landmarks
        #fixed_landmarks = fixed_landmarks[0,:][None,...]
        
        fcoords, ccoords = torch.floor(fixed_landmarks).long(), torch.ceil(fixed_landmarks).long()

        #enforce that coords must fall inside displacement grid
        u = torch.tensor([[displacement_field.shape[2]-1, displacement_field.shape[3]-1, displacement_field.shape[4]-1]]).to(displacement_field.device)
        l = torch.tensor([[0, 0, 0]]).to(displacement_field.device)
        fcoords = torch.max(torch.min(fcoords, u), l)
        ccoords = torch.max(torch.min(ccoords, u), l)

        #print(ccoords.max())
        f_displacements = displacement_field[:,:,fcoords[:,0], fcoords[:,1], fcoords[:,2]]
        c_displacements = displacement_field[:,:,ccoords[:,0], ccoords[:,1], ccoords[:,2]]
        displacements = (f_displacements + c_displacements)/2
        # assert displacements.requires_grad
        displacements = einops.rearrange(displacements, 'b n N -> (b N) n')
        return torch.linalg.norm((fixed_landmarks + displacements-moving_landmarks)*moving_spacing, dim=1).mean()

class MINDLoss(nn.Module):
    """
    Computes a MIND feature space loss of the input images

    Parameters
    ----------
    radius: int
        Radius for MIND SSC features. Default=1
    dilation: int
        Dilation for MIND SSC features. Default=2
    """
    def __init__(self, radius: int = 1, dilation: int = 2):
        super().__init__()
        self.mind = partial(MINDSSC, radius=radius, dilation=dilation)

    def forward(self, im1: torch.Tensor, im2: torch.Tensor) -> torch.Tensor:
        im1, im2 = [einops.repeat(im.squeeze(), 'd h w -> b c d h w', b=1, c=1) for im in (im1, im2)]
        mind1 = self.mind(im1)
        mind2 = self.mind(im2)
        loss = F.mse_loss(mind1, mind2)
        return loss


class Grad(nn.Module):
    """
    N-D gradient loss. Modified from Voxelmorph.

    Parameters
    ----------
    penalty: str
        Either l1 or l2. Default=l1
    """

    def __init__(self, penalty='l1'):
        super().__init__()
        if penalty not in ('l1' or 'l2'):
            raise ValueError("Penalty must either be l1 or l2")
        self.penalty = penalty

    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        return grad

class NCC(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super().__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true.squeeze().unsqueeze(0).unsqueeze(0)
        Ji = y_true.squeeze().unsqueeze(0).unsqueeze(0)

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

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

        return -torch.mean(cc)


class MSE(nn.Module):
    """
    Mean squared error loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

if __name__ == "__main__":
    from common import load_keypoints
    from pathlib import Path
    import math
    import typer

    app = typer.Typer()

    @app.command()
    def load_all(keypoints_dir: Path):
        for keypoint_csv in keypoints_dir.iterdir():
            load_keypoints(keypoint_csv)

    @app.command()
    def main(keypoints_csv: Path):
        keypoints = load_keypoints(keypoints_csv) 
        disp_shape = [math.ceil(torch.max(keypoints[:,i]).item())+1 for i in range(keypoints.shape[-1])]
        disp = torch.rand((1, 3, *disp_shape))
        disp.requires_grad = True
        tre = TotalRegistrationLoss()(
            fixed_landmarks=keypoints,
            moving_landmarks=keypoints,
            displacement_field= disp,
            fixed_spacing= torch.randn((1,3)),
            moving_spacing= torch.randn((1,3)),
        )


    app()
