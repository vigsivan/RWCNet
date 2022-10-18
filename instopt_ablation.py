from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import nibabel as nib

import os
import json
import typer
import einops
import numpy as np

import torch
import torch.nn.functional as F

from common import MINDSSC, get_identity_affine_grid, apply_displacement_field, get_labels
from metrics import compute_total_registration_error, compute_dice
from instopt_loops_ablation import inst_optimization
from ioa_swa_loops import swa_optimization

app = typer.Typer()

get_spacing = lambda x: np.sqrt(np.sum(x.affine[:3, :3] * x.affine[:3, :3], axis=0))
add_bc_dim = lambda x: einops.repeat(x, "d h w -> b c d h w", b=1, c=1).float()

GPU_iden = 0
torch.cuda.set_device(GPU_iden)

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
    moving_name: Optional[str]


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
        moving_name = f"disp_{img_number}_0001.nii.gz"
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
            moving_name=moving_name,
        )

k_path = "/home/tvujovic/repos/instance_opt/optimization-based-registration/outputs/rnn_45000/disps/"
rnn_path = './eval_0929/'

@app.command()
def apply_instance_optimization(
        data_json: Path = Path("OASIS_0711.json"),
        initial_disp_root: Path = Path(k_path),
        save_directory: Path = Path("./oasis_start_noise/"),
        split_val: str = "train",
        half_res: bool = False,
        use_mask: bool = True,
):

    save_directory.mkdir(exist_ok=True)
    (save_directory / "disps").mkdir(exist_ok=True)
    device = "cuda"

    checkpoint_dir = save_directory / 'opt_checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    measurements = defaultdict(dict)

    img_shape = (80, 112, 96)
    grid0 = get_identity_affine_grid(img_shape)

    gen = tqdm(get_paths(data_json=data_json, split_val=split_val, disp_root=initial_disp_root))

    for data in gen:
        # gaussian_noise = (0.9 ** 0.5) * torch.randn((160, 224, 192)).to(device)
        image_name = data.fixed_image.name[-16:-12]

        fixed_nib = nib.load(data.fixed_image)
        moving_nib = nib.load(data.moving_image)
        # initial_disp_nib = nib.load(data.rnn_disp_path)

        fixed = add_bc_dim(torch.from_numpy(fixed_nib.get_fdata())).to(device).squeeze()
        moving = add_bc_dim(torch.from_numpy(moving_nib.get_fdata())).to(device).squeeze()

        if data.fixed_keypoints is not None:
            fixed_keypoints = np.loadtxt(data.fixed_keypoints, delimiter=",")
            moving_keypoints = np.loadtxt(data.moving_keypoints, delimiter=",")
            fs = get_spacing(fixed_nib)

        if data.fixed_mask is not None:
            fixed_mask = torch.from_numpy(nib.load(data.fixed_mask).get_fdata().astype('float32')).to(device)
            moving_mask = torch.from_numpy(nib.load(data.moving_mask).get_fdata().astype('float32')).to(device)

        if use_mask and (data.fixed_mask is not None):
            fixed = fixed_mask * fixed
            moving = moving_mask * moving

        if data.fixed_segmentation is not None:
            fixed_seg = (nib.load(data.fixed_segmentation)).get_fdata().astype("float")
            moving_seg = (nib.load(data.moving_segmentation)).get_fdata().astype("float")

            label_list = get_labels(torch.tensor(fixed_seg), torch.tensor(moving_seg))

            fixed_segt = torch.tensor(fixed_seg).to(device).unsqueeze(0).unsqueeze(0)
            moving_segt = torch.tensor(moving_seg).to(device).unsqueeze(0).unsqueeze(0)

        # disp_rnn = initial_disp_nib.get_fdata()
        # disp_torch = torch.from_numpy(einops.rearrange(disp_rnn, 'h w d N -> N h w d')).unsqueeze(0).to(device)

        fixed = (fixed - fixed.min())/(fixed.max() - fixed.min())
        moving = (moving - moving.min())/(moving.max() - moving.min())

        mindssc_fix_ = MINDSSC(fixed.unsqueeze(0).unsqueeze(0), 1, 2).half()
        mindssc_mov_ = MINDSSC(moving.unsqueeze(0).unsqueeze(0), 1, 2).half()

        grid_sp = 2
        with torch.no_grad():
            mind_fix_ = F.avg_pool3d(mindssc_fix_, grid_sp, stride=grid_sp)
            mind_mov_ = F.avg_pool3d(mindssc_mov_, grid_sp, stride=grid_sp)

        if half_res:
            fixed_segt = F.avg_pool3d(fixed_segt, grid_sp, stride=grid_sp)
            moving_segt = F.avg_pool3d(moving_segt, grid_sp, stride=grid_sp)

        net = inst_optimization(
            disp=None,
            mind_fixed=mind_fix_,
            mind_moving=mind_mov_,
            lambda_weight=1.25,
            image_shape=tuple(s//grid_sp for s in fixed_nib.shape),
            norm=grid_sp,
            img_name=image_name, fkp=None, mkp=None,
            fs=None, checkpoint_dir=checkpoint_dir, grid0=grid0,
            fsegt=fixed_segt,
            msegt=moving_segt,
            labels=None,
        )

        disp_sample = F.avg_pool3d(
            F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1
        ).permute(0, 2, 3, 4, 1)

        scale = (torch.tensor([(160 - 1) / 2, (224 - 1) / 2, (192 - 1) / 2, ]).cuda().unsqueeze(0))
        grid_disp = (grid0.view(-1, 3).cuda().float()
                     + ((disp_sample.view(-1, 3)) / scale).flip(1).float())

        patch_mov_sampled = F.grid_sample(
            mind_mov_.float(),
            grid_disp.view(1, 80, 112, 96, 3).cuda(),
            align_corners=False,
            mode="bilinear",
            padding_mode="border")

        sampled_cost = (patch_mov_sampled - mind_fix_).pow(2).mean(1) * 12
        loss = sampled_cost.mean().item()
        measurements[data.disp_name]["img_loss"] = loss

        fitted_grid = disp_sample.permute(0, 4, 1, 2, 3).detach()
        disp_hr = F.interpolate(
            fitted_grid * grid_sp,
            size=fixed_nib.shape,
            mode="trilinear",
            align_corners=False,
        )

        disp_np = disp_hr.squeeze(0).detach().cpu().numpy()

        l2r_disp = einops.rearrange(disp_np.squeeze(), 't h w d -> h w d t')
        new_disp_path = save_directory / "disps" / data.disp_name
        new_disp_path.parent.mkdir(exist_ok=True)

        if data.fixed_segmentation is not None:
            moved_seg = apply_displacement_field(disp_np, moving_seg, order=0)

            dice = compute_dice(
                fixed_seg, moving_seg, moved_seg, labels=label_list #type: ignore
            )
            measurements[data.disp_name]["dice"] = dice
            print(dice)

        # if data.fixed_keypoints is not None:
        #
        #     tre_np = compute_total_registration_error(
        #         fixed_keypoints, moving_keypoints, l2r_disp, fs, fs
        #     )
        #
        #     fixed_keypoints = torch.from_numpy(fixed_keypoints)
        #     moving_keypoints = torch.from_numpy(moving_keypoints)
        #
        #     measurements[data.disp_name]["tre"] = tre_np
        #     print(tre_np)

        # displacement_nib = nib.Nifti1Image(l2r_disp, affine=moving_nib.affine)
        # nib.save(displacement_nib, new_disp_path)
        #
        # moved_mask = apply_displacement_field(disp_np, nib.load(data.moving_mask).get_fdata())
        # mask_nib = nib.Nifti1Image(moved_mask, affine=fixed_nib.affine)
        # nib.save(mask_nib, save_directory / "masks" / data.moving_name)
        #
        # moved_image = apply_displacement_field(disp_np, moving_nib.get_fdata())
        # moved_nib = nib.Nifti1Image(moved_image, affine=fixed_nib.affine)
        # nib.save(moved_nib, save_directory / "images" / (data.moving_name))

    with open(save_directory / "measurements.json", "w") as f:
        json.dump(measurements, f)

if __name__ == "__main__":
    app()
