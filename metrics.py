"""
Metrics stolen from the evaluation script
"""

from typing import List
import numpy as np
from scipy.ndimage import map_coordinates, correlate
from surface_distance import compute_surface_distances, compute_robust_hausdorff


def compute_log_jacobian_determinant_standard_deviation(
    disp: np.ndarray,
):
    # TODO: what is this business about the mask?
    jac_det = (
        jacobian_determinant(disp[np.newaxis, :, :, :, :].transpose((0, 4, 1, 2, 3)))
        + 3
    ).clip(0.000000001, 1000000000)
    log_jac_det = np.log(jac_det)
    return log_jac_det.std()


def jacobian_determinant(disp: np.ndarray):
    assert len(disp.shape) == 5

    gradx = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack(
        [
            correlate(disp[:, 0, :, :, :], gradx, mode="constant", cval=0.0),
            correlate(disp[:, 1, :, :, :], gradx, mode="constant", cval=0.0),
            correlate(disp[:, 2, :, :, :], gradx, mode="constant", cval=0.0),
        ],
        axis=1,
    )

    grady_disp = np.stack(
        [
            correlate(disp[:, 0, :, :, :], grady, mode="constant", cval=0.0),
            correlate(disp[:, 1, :, :, :], grady, mode="constant", cval=0.0),
            correlate(disp[:, 2, :, :, :], grady, mode="constant", cval=0.0),
        ],
        axis=1,
    )

    gradz_disp = np.stack(
        [
            correlate(disp[:, 0, :, :, :], gradz, mode="constant", cval=0.0),
            correlate(disp[:, 1, :, :, :], gradz, mode="constant", cval=0.0),
            correlate(disp[:, 2, :, :, :], gradz, mode="constant", cval=0.0),
        ],
        axis=1,
    )

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = (
        jacobian[0, 0, :, :, :]
        * (
            jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :]
            - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]
        )
        - jacobian[1, 0, :, :, :]
        * (
            jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :]
            - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]
        )
        + jacobian[2, 0, :, :, :]
        * (
            jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :]
            - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :]
        )
    )

    return jacdet


def compute_hd95(
    fixed: np.ndarray, moving: np.ndarray, moving_warped: np.ndarray, labels
):
    fixed, moving, moving_warped = (
        fixed.squeeze(),
        moving.squeeze(),
        moving_warped.squeeze(),
    )
    hd95 = 0
    count = 0
    for i in labels:
        if ((fixed == i).sum() == 0) or ((moving == i).sum() == 0):
            continue
        hd95 += compute_robust_hausdorff(
            compute_surface_distances((fixed == i), (moving_warped == i), np.ones(3)),
            95.0,
        )
        count += 1
    hd95 /= count
    return hd95


def compute_total_registration_error(
    fix_lms: np.ndarray,
    mov_lms: np.ndarray,
    disp: np.ndarray,
    spacing_fix: np.ndarray,
    spacing_mov: np.ndarray,
) -> float:
    """
    Computes the total registratin error from keypoints.

    Parameters
    ----------
    fix_lms: np.ndarray
        Keypoints from the fixed image
    mov_lms: np.ndarray
        Keypoints from the moving image
    disp: np.ndarray
        Displacement field
    spacing_fix: np.ndarray
        Voxel spacing of the fixed image
    spacing_mov: np.ndarray
        Voxel spacing of the moving image

    Returns
    -------
    total_registration_error: float
    """
    #fix_lms_tmp = fix_lms
    #fix_lms = fix_lms[0,:][None,...]
    #mov_lms = mov_lms[0,:][None,...]
    fix_lms_disp_x = map_coordinates(disp[:, :, :, 0], fix_lms.transpose())
    fix_lms_disp_y = map_coordinates(disp[:, :, :, 1], fix_lms.transpose())
    fix_lms_disp_z = map_coordinates(disp[:, :, :, 2], fix_lms.transpose())
    fix_lms_disp = np.array((fix_lms_disp_x, fix_lms_disp_y, fix_lms_disp_z)).transpose()

    fix_lms_warped = fix_lms + fix_lms_disp

    # fixed_landmarks = (fixed_landmarks).to(displacement_field.device)
    # moving_landmarks = (moving_landmarks).to(displacement_field.device)
    # moving_spacing = (moving_spacing).to(displacement_field.device)
    #
    # assert fixed_landmarks.shape == moving_landmarks.shape
    # fcoords, ccoords = torch.floor(moving_landmarks).long(), torch.ceil(moving_landmarks).long()
    # f_displacements = displacement_field[:, :, fcoords[:, 0], fcoords[:, 1], fcoords[:, 2]]
    # c_displacements = displacement_field[:, :, ccoords[:, 0], ccoords[:, 1], ccoords[:, 2]]
    # displacements = (f_displacements + c_displacements) / 2

    tre = np.linalg.norm((fix_lms_warped - mov_lms) * spacing_mov, axis=1)

    return tre.mean().item()


def compute_dice(
    fixed: np.ndarray, moving: np.ndarray, moving_warped: np.ndarray, labels: List[int]
) -> float:
    dice = 0
    count = 0
    for i in labels:
        if ((fixed == i).sum() == 0) or ((moving == i).sum() == 0):
            continue
        computed_dice = _compute_dice_coefficient((fixed == i), (moving_warped == i))
        dice += computed_dice
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
    return 2 * volume_intersect / volume_sum
