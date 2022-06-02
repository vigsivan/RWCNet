"""
Metrics stolen from the evaluation script
"""

from typing import List
import numpy as np

def compute_dice(fixed: np.ndarray,moving: np.ndarray,moving_warped: np.ndarray, labels: List[int]):
    dice = 0
    count = 0
    for i in labels:
        if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
            continue
        dice += _compute_dice_coefficient((fixed==i), (moving_warped==i))
        count += 1
    dice /= count
    return dice

def _compute_dice_coefficient(mask_gt: np.ndarray, mask_pred: np.ndarray):
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
    return 0
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum
