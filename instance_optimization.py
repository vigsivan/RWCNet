from typing import Tuple
from pathlib import Path
import typer
import einops
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm
from common import adam_optimization, MINDSSC

app = typer.Typer()

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

@app.command()
def apply_instance_optimization(data_root: Path, disp_root: Path, save_directory: Path, ext: str=".nii.gz", half_res: bool=True):
    files = [i for i in disp_root.iterdir() if i.name.endswith(ext)]
    add_bc_dim = lambda x: einops.repeat(x, "d h w -> b c d h w", b=1, c=1).float()
    device="cuda"
    save_directory.mkdir(exist_ok=True)
    for f in tqdm(files):
        disp_nib = nib.load(f) 
        disp_sk = disp_nib.get_fdata()
        disp_torch = torch.from_numpy(einops.rearrange(disp_sk, 'h w d N -> N h w d')).unsqueeze(0).to(device)
        fixed_image, moving_image = load_corresponding_files(data_root, f.name)

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


        if half_res:
            shape = [s//2 for s in disp_torch.shape[-3:]]
            disp_torch = F.interpolate(disp_torch, shape)


        net = adam_optimization(
            disp=disp_torch,
            mind_fixed=mind_fix_,
            mind_moving=mind_mov_,
            lambda_weight=1.5,
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

        disp_np = disp_hr.detach().cpu().numpy()

        # NOTE: we are using scipy's interpolate func, which does not take a batch dimension
        l2r_disp = einops.rearrange(disp_np.squeeze(), 't h w d -> h w d t')
        disp_name = f"disp_{fixed_image.name[-16:-12]}_{moving_image.name[-16:-12]}"
        displacement_nib = nib.Nifti1Image(l2r_disp, affine=moving_nib.affine)
        nib.save(displacement_nib, save_directory / f"{disp_name}.nii.gz")

app()
