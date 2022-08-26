import json
from collections import defaultdict
from typing import Tuple
from pathlib import Path
import typer
import einops
import nibabel as nib
import torch
from torch import nn
import random
import torch.nn.functional as F
from tqdm import tqdm, trange
from differentiable_metrics import MutualInformationLoss, TotalRegistrationLoss
from common import adam_optimization, MINDSSC, adam_optimization_teo, load_keypoints, swa_optimization, warp_image
import numpy as np

app = typer.Typer()

get_spacing = lambda x: np.sqrt(np.sum(x.affine[:3, :3] * x.affine[:3, :3], axis=0))

upsamplenet = nn.Sequential(
    nn.Conv3d(3,3,3,1,padding='same'),
    nn.ReLU(),
    nn.Conv3d(3,3,3,1,padding='same'),
)

def load_corresponding_files(files_path: Path, disp_name: str) -> Tuple[Path, Path]:
    file_names = list(files_path.iterdir()) 
    _, fixed_id, moving_id = disp_name.split(".")[0].split("_")
    fixed_image, moving_image = None, None
    for f in file_names:
        if f.name[-16:-12] == fixed_id:
            if f.name.split(".")[0].split("_")[-1] == "0000":
                fixed_image = f
            else:
                moving_image = f

    if fixed_image is None or moving_image is None:
        raise Exception

    return fixed_image, moving_image

def load_corresponding_kps(files_path: Path, disp_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    files_path = Path("/home/vsivan/scratch/NLST/keypointsTr/")
    file_names = list(files_path.iterdir())
    _, fixed_id, moving_id = disp_name.split(".")[0].split("_")
    fixed_image, moving_image = None, None
    for f in file_names:
        if f.name[-16:-12] == fixed_id:
            if f.name.split(".")[0].split("_")[-1] == "0000":
                fixed_image = f
            else:
                moving_image = f

    if fixed_image is None or moving_image is None:
        raise Exception

    return load_keypoints(fixed_image), load_keypoints(moving_image)


@app.command()
def apply_instance_optimization(data_root: Path, disp_root: Path, save_directory: Path, ext: str=".nii.gz", half_res: bool=True):
    # files = [i for i in disp_root.iterdir() if i.name.endswith(ext)]
    files = [i for i in disp_root.iterdir()]
    add_bc_dim = lambda x: einops.repeat(x, "d h w -> b c d h w", b=1, c=1).float()
    device="cuda"
    save_directory.mkdir(exist_ok=True)
    mind = MutualInformationLoss()
    trl =TotalRegistrationLoss()
    losses = defaultdict(dict)
    kplosses = defaultdict(dict)
    upsample = upsamplenet.to(device)
    opt = torch.optim.Adam(upsample.parameters(), lr=.01)
    random.shuffle(files)
    for i, f in tqdm(enumerate(files)):
        # disp_torch = torch.load(f)
        # disp_torch =  disp_torch.to(device)
        # if disp_torch.shape[-1] == 112:
        #     print(f"skipping {f}")
        #     continue
        # breakpoint()
        disp_nib = nib.load(f)
        disp_sk = disp_nib.get_fdata()
        disp_torch = torch.from_numpy(einops.rearrange(disp_sk, 'h w d N -> N h w d')).unsqueeze(0).to(device)
        fixed_image, moving_image = load_corresponding_files(data_root, f.name.split('.')[0] + ".nii.gz")
        # fkps, mkps = load_corresponding_kps(data_root, f.name.split('.')[0] + ".nii.gz")

        fixed_nib = nib.load(fixed_image)
        moving_nib = nib.load(moving_image)

        fixed = add_bc_dim(torch.from_numpy(fixed_nib.get_fdata())).to(device)
        moving = add_bc_dim(torch.from_numpy(moving_nib.get_fdata())).to(device)

        fixed = (fixed - fixed.min())/(fixed.max() - fixed.min())
        moving = (moving - moving.min())/(moving.max() - moving.min())

        mindssc_fix_ = MINDSSC(
            fixed.cuda(), 1, 2
        ).half()
        mindssc_mov_ = MINDSSC(
            moving.cuda(), 1, 2
        ).half()

        grid_sp = 2
        mind_fix_ = F.avg_pool3d(mindssc_fix_, grid_sp, stride=grid_sp)
        mind_mov_ = F.avg_pool3d(mindssc_mov_, grid_sp, stride=grid_sp)

        out = warp_image(disp_torch.float(), moving)
        before = mind(out, fixed)
        # b4kp = trl(fkps, mkps, disp_torch, torch.from_numpy(get_spacing(fixed_nib)), torch.from_numpy(get_spacing(moving_nib)))

        losses[f.name]["before"] = before.item()

        if half_res:
            shape = [s//2 for s in disp_torch.shape[-3:]]
            disp_torch = F.interpolate(disp_torch, shape)


        net = adam_optimization_teo(
            disp=disp_torch,
            mind_fixed=mind_fix_,
            mind_moving=mind_mov_,
            lambda_weight=0.5,
            image_shape=tuple(s//grid_sp for s in fixed_nib.shape),
            iterations=100,
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
        disp_hr = disp_hr.detach()

        after = mind(warp_image(disp_hr, moving), fixed)
        losses[f.name]["after"] = after.item()

        net = adam_optimization_teo(
            disp=disp_hr,
            mind_fixed=mindssc_fix_,
            mind_moving=mindssc_mov_,
            lambda_weight=0.5,
            image_shape=tuple(s//1 for s in fixed_nib.shape),
            iterations=100,
            norm=1
        )

        disp_sample = F.avg_pool3d(
            F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1
        ).permute(0, 2, 3, 4, 1)
        disp_hr = disp_sample.permute(0, 4, 1, 2, 3).detach()

        after = mind(warp_image(disp_hr, moving), fixed)

        losses[f.name]["after2"] = after.item()
        disp_np = disp_hr.detach().cpu().numpy()

        # NOTE: we are using scipy's interpolate func, which does not take a batch dimension
        l2r_disp = einops.rearrange(disp_np.squeeze(), 't h w d -> h w d t')
        disp_name = f"disp_{fixed_image.name[-16:-12]}_{moving_image.name[-16:-12]}"
        displacement_nib = nib.Nifti1Image(l2r_disp, affine=moving_nib.affine)
        nib.save(displacement_nib, save_directory / f"{disp_name}.nii.gz")
    with open(save_directory/"losses_teo.json", 'w') as fp:
        json.dump(losses, fp)

app()
