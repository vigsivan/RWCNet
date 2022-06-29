"""
Implements differentiable loss functions for training
"""

from functools import partial
from typing import List

import einops
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
# from monai.losses.dice import DiceLoss
from monai.losses.image_dissimilarity import GlobalMutualInformationLoss as MutualInformationLoss
from common import MINDSSC, identity_grid_torch

__all__ = ["DiceLoss", "MutualInformationLoss", "T torch.Tensor torch.TensorotalRegistrationLoss", "Grad"]

class DiceLoss(nn.Module):
    def __init__(self, labels: List[int]=[1]):
        super().__init__()
        self.labels = labels

    def forward(self, fixed: torch.Tensor, moving: torch.Tensor, moving_warped: torch.Tensor):
        dice = 0
        count = 0
        for i in self.labels:
            if ((fixed == i).sum() == 0) or ((moving == i).sum() == 0):
                continue
            dice += _compute_dice_coefficient((fixed == i), (moving_warped == i))
            count += 1
        dice /= count
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
        return torch.Tensor(0)
    volume_intersect = (mask_gt & mask_pred).sum()
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
        assert fixed_landmarks.shape == moving_landmarks.shape
        displacements = displacement_field[:,:,moving_landmarks[:,0], moving_landmarks[:,1], moving_landmarks[:,2]]
        assert displacements.requires_grad
        displacements = einops.rearrange(displacements, 'b n N -> (b N) n')
        return (moving_landmarks + displacements-fixed_landmarks)*moving_spacing

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

if __name__ == "__main__":
    from common import load_keypoints
    from pathlib import Path
    import math
    import typer

    app = typer.Typer()

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
