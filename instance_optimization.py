from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from tqdm import tqdm

import nibabel as nib

import os
import json
import typer
import einops
import numpy as np

import torch
import torch.nn.functional as F

from common import MINDSSC
from optimizer_loops import swa_optimization

app = typer.Typer()

get_spacing = lambda x: np.sqrt(np.sum(x.affine[:3, :3] * x.affine[:3, :3], axis=0))
add_bc_dim = lambda x: einops.repeat(x, "d h w -> b c d h w", b=1, c=1).float()


@dataclass
class InstanceOptData:
    fixed_image: Path
    moving_image: Path
    fixed_mask: Optional[Path]
    moving_mask: Optional[Path]
    fixed_segmentation: Optional[Path]
    moving_segmentation: Optional[Path]
    fixed_keypoints: Optional[Path]
    moving_keypoints: Optional[Path]
    fixed_landmarks: Optional[Path]
    moving_landmarks: Optional[Path]
    rnn_disp_path: Optional[Path]
    disp_name: Optional[str]


def get_paths(
        data_json,
        split_val,
        disp_root,
):
    with open(data_json, "r") as f:
        data = json.load(f)[split_val]

    has_segs = "fixed_segmentation" in data[0]
    has_kps = "fixed_keypoints" in data[0]
    has_mask = "fixed_mask" in data[0]
    has_lms = "fixed_landmarks" in data[0]

    for v in data:
        img_number = v['moving_image'][-16:-12]
        disp_name = f"disp_{img_number}_{img_number}.nii.gz"
        disp_path = disp_root / disp_name

        yield InstanceOptData(
            fixed_image=Path(v["fixed_image"]),
            moving_image=Path(v["moving_image"]),
            fixed_mask=Path(v["fixed_mask"]) if has_mask else None,
            moving_mask=Path(v["moving_mask"]) if has_mask else None,
            fixed_segmentation=Path(v["fixed_segmentation"]) if has_segs else None,
            moving_segmentation=Path(v["moving_segmentation"]) if has_segs else None,
            fixed_keypoints=Path(v["fixed_keypoints"]) if has_kps else None,
            moving_keypoints=Path(v["moving_keypoints"]) if has_kps else None,
            fixed_landmarks=Path(v["fixed_landmarks"]) if has_lms else None,
            moving_landmarks=Path(v["fixed_landmarks"]) if has_lms else None,
            rnn_disp_path=Path(disp_path),
            disp_name=disp_name,
        )


@app.command()
def apply_instance_optimization(
        data_json: Path,
        initial_disp_root: Path,
        save_directory: Path,
        split_val,
        half_res: bool=True,
        use_mask: bool=True,
):

    save_directory.mkdir(exist_ok=True)
    (save_directory / "disps").mkdir(exist_ok=True)
    device = "cuda"

    gen = tqdm(get_paths(data_json=data_json, split_val=split_val, disp_root=initial_disp_root))

    for data in gen:

        fixed_nib = nib.load(data.fixed_image)
        moving_nib = nib.load(data.moving_image)
        initial_disp_nib = nib.load(data.rnn_disp_path)

        fixed = add_bc_dim(torch.from_numpy(fixed_nib.get_fdata())).to(device).squeeze()
        moving = add_bc_dim(torch.from_numpy(moving_nib.get_fdata())).to(device).squeeze()

        # Un-comment if we decide to use any of these during instance opt
        # if data.fixed_keypoints is not None:
        #     fixed_keypoints = np.loadtxt(data.fixed_keypoints, delimiter=",")
        #     moving_keypoints = np.loadtxt(data.moving_keypoints, delimiter=",")
        #
        # if data.fixed_segmentation is not None:
        #     fixed_segmentation =np.loadtxt(data.fixed_segmentation, delimiter=",")
        #     moving_segmentation = np.loadtxt(data.moving_segmentation, delimiter=",")
        #
        # if data.fixed_landmarks is not None:
        #     fixed_landmarks =np.loadtxt(data.fixed_landmarks, delimiter=",")
        #     moving_landmarks = np.loadtxt(data.moving_landmarks, delimiter=",")

        if data.fixed_mask is not None:
            fixed_mask = torch.from_numpy(nib.load(data.fixed_mask).get_fdata().astype('float32')).to(device)
            moving_mask = torch.from_numpy(nib.load(data.moving_mask).get_fdata().astype('float32')).to(device)

        if use_mask and (data.fixed_mask is not None):
            fixed = fixed_mask * fixed
            moving = moving_mask * moving

        disp_rnn = initial_disp_nib.get_fdata()
        disp_torch = torch.from_numpy(einops.rearrange(disp_rnn, 'h w d N -> N h w d')).unsqueeze(0).to(device)

        fixed = (fixed - fixed.min())/(fixed.max() - fixed.min())
        moving = (moving - moving.min())/(moving.max() - moving.min())

        mindssc_fix_ = MINDSSC(fixed.unsqueeze(0).unsqueeze(0), 1, 2).half()
        mindssc_mov_ = MINDSSC(moving.unsqueeze(0).unsqueeze(0), 1, 2).half()

        grid_sp = 2
        with torch.no_grad():
            mind_fix_ = F.avg_pool3d(mindssc_fix_, grid_sp, stride=grid_sp)
            mind_mov_ = F.avg_pool3d(mindssc_mov_, grid_sp, stride=grid_sp)

        if half_res:
            shape = [s//2 for s in disp_torch.shape[-3:]]
            disp_torch = F.interpolate(disp_torch, shape)

        net = swa_optimization(
            disp=disp_torch,
            mind_fixed=mind_fix_,
            mind_moving=mind_mov_,
            lambda_weight=1.25,
            image_shape=tuple(s//grid_sp for s in fixed_nib.shape),
            norm=grid_sp
        )

        disp_sample = F.avg_pool3d(
            F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1
        ).permute(0, 2, 3, 4, 1)

        fitted_grid = disp_sample.permute(0, 4, 1, 2, 3).detach()
        disp_hr = F.interpolate(
            fitted_grid * grid_sp,
            size=fixed_nib.shape,
            mode="trilinear",
            align_corners=False,
        )

        disp_np = disp_hr.detach().cpu().numpy()

        l2r_disp = einops.rearrange(disp_np.squeeze(), 't h w d -> h w d t')
        new_disp_path = save_directory / "disps" / data.disp_name
        new_disp_path.parent.mkdir(exist_ok=True)

        displacement_nib = nib.Nifti1Image(l2r_disp, affine=moving_nib.affine)
        nib.save(displacement_nib, new_disp_path)


if __name__ == "__main__":
    app()
