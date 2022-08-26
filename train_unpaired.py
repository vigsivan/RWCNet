import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import einops
import pickle
import numpy as np
from typing import Dict, Optional, Tuple, Union
from torch.utils.tensorboard.writer import SummaryWriter
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from typer import Typer
from tqdm import trange, tqdm

from common import (
    concat_flow,
    identity_grid_torch,
    load_keypoints,
    tb_log,
    torch2skimage_disp,
    warp_image,
)
from differentiable_metrics import (
    MINDLoss,
    DiceLoss,
    Grad,
    TotalRegistrationLoss,
    MutualInformationLoss,
)
from train import PatchDataset
from networks import SomeNet

app1 = Typer()


get_spacing = lambda x: np.sqrt(np.sum(x.affine[:3, :3] * x.affine[:3, :3], axis=0))


@contextmanager
def evaluating(net):
    """Temporarily switch to evaluation mode."""
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()

class ImageDataset(Dataset):
    """
    Loads images
    """

    def __init__(
        self,
        data_json: Path,
        split: str,
        downsample: int
    ):
        with open(data_json, "r") as f:
            data = json.load(f)[split]

        self.data = data
        self.downsample = downsample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        image_nib = nib.load(data["image"])
        image = torch.from_numpy(image_nib.get_fdata())

        rshape = tuple(i // self.downsample for i in image.shape[-3:])

        image = (image - image.min())/(image.max()-image.min())
        image = image.squeeze().unsqueeze(0)
        image = F.interpolate(image.unsqueeze(0), rshape).squeeze(0).float()

        ret = {"image": image}

        if "segmentation" in data:
            segmentation_nib = nib.load(data["segmentation"])
            segmentation = torch.from_numpy(segmentation_nib.get_fdata()).float()
            ret["segmentation"] = segmentatation

        if "mask" in data:
            mask_nib = nib.load(data["mask"])
            mask = torch.from_numpy(mask_nib.get_fdata()).float()
            ret["mask"] = mask

        return ret

@app1.command()
def train_stage2(
    data_json: Path,
    stage1_downsample: int,
    stage1_model: Path,
    stage2_downsample: int,
    stage2_patchfactor: int,
    checkpoint_dir: Path,
    start: Optional[Path]= None,
    device: str = "cuda",
    iters: int = 12,
    search_range: int = 3,
    steps: int=50000,
    lr: float=3e-4,
    image_loss_weight: float=1,
    reg_loss_weight: float=.01,
    log_freq: int=1,
    val_freq: int=1,
    save_freq: int=1,
    stage1_model_iters: int=12,
    stage1_model_search_range: int=3
):
    train_dataset = ImageDataset(data_json, split="train", downsample=stage1_downsample)
    val_dataset = PatchDataset(data_json, split="val", res_factor=stage1_downsample, patch_factor=stage1_downsample)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True) #, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True) #, num_workers=4)

    r, p = 1 / stage2_downsample, 1 / stage2_patchfactor
    n_patches = (((r - p) / p) + 1) ** 2
    chan_split = r / p

    model_stage1 = SomeNet(iters=stage1_model_iters, search_range=stage1_model_search_range).to(device)
    model_stage1 = model_stage1.eval().to(device)
    model_stage1.requires_grad = False

    model = SomeNet(iters=iters, search_range=search_range)

    step_count = 0
    if start is not None:
        model.load_state_dict(torch.load(start))
        step_count = int(start.name.split("_")[-1].split(".")[0])

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=checkpoint_dir)

    print(f"Dataset size is {len(train_dataset)}")
    print(f"Starting training from step {step_count}")

    while step_count + 1 < steps:
        for step_count, data in zip(trange(step_count, steps), train_loader):
            fixed = data["image"][0,...].unsqueeze(0)
            moving = data["image"][1,...].unsqueeze(0)
            fixed, moving = fixed.to(device), moving.to(device)
            flow, hidden = model(fixed, moving)

            moved = warp_image(flow, moving)

            losses_dict: Dict[str, torch.Tensor] = {}

            losses_dict["image_loss"] = image_loss_weight * MutualInformationLoss()(
                moved.squeeze(), fixed.squeeze()
            )
            losses_dict["grad"] = reg_loss_weight * Grad()(flow)

            if "segmentation" in data:
                fixed_segmentation = data["segmentataion"][0,...].unsqueeze(0)
                moving_segmentation = data["segmentataion"][1,...].unsqueeze(0)

                moved_segmentation = warp_image(flow, moving_mask)

                fixed_segmentation = torch.round(fixed_segmentation)
                moved_segmentation = torch.round(moved_segmentation)

                losses_dict["dice_loss"] = seg_loss_weight * DiceLoss()(
                    fixed_segmenatation, moved_segmentation())

            total_loss = sum(losses_dict.values())
            assert isinstance(total_loss, torch.Tensor)
            losses_dict_log = {k: v.item() for k, v in losses_dict.items()}

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            if step_count % log_freq == 0:
                tb_log(
                    writer,
                    losses_dict_log,
                    step=step_count,
                    moving_fixed_moved=(moving, fixed, moved),
                )

            if val_freq > 0 and step_count % val_freq == 0 and step_count > 0:
                losses_cum_dict = defaultdict(list)
                with torch.no_grad(), evaluating(model):
                    for data in val_loader:
                        fixed, moving = data["fixed_image"], data["moving_image"]
                        fixed, moving = fixed.to(device), moving.to(device)
                        flow, hidden = model(fixed, moving)

                        moved = warp_image(flow, moving)

                        losses_cum_dict["image_loss"].append(
                            (
                                image_loss_weight
                                * MutualInformationLoss()(
                                    moved.squeeze(), fixed.squeeze()
                                )
                            ).item()
                        )
                        losses_cum_dict["grad"].append(
                            (reg_loss_weight * Grad()(flow)).item()
                        )

                        if "fixed_segmentation" in data:
                            fixed_segmentation = data["fixed_segmentation"].to(device).float()
                            moving_segmentation = data["moving_segmentation"].to(device).float()

                            moved_segmentation = warp_image(flow, moving_mask)

                            fixed_segmentation = torch.round(fixed_segmentation)
                            moved_segmentation = torch.round(moved_segmentation)

                            losses_cum_dict["dice_loss"].append(
                                    seg_loss_weight
                                    * DiceLoss()(fixed_segmenatation, moved_segmentation())
                            )


                for k, v in losses_cum_dict.items():
                    writer.add_scalar(
                        f"val_{k}", np.mean(v).item(), global_step=step_count
                    )

            if step_count % save_freq == 0:
                torch.save(
                    model.state_dict(),
                    checkpoint_dir / f"rnn{stage1_downsample}x_{step_count}.pth",
                )

    torch.save(model.state_dict(), checkpoint_dir / f"rnn{stage1_downsample}x_{step_count}.pth")


@app1.command()
def train_stage1(
    data_json: Path,
    stage1_downsample: int,
    checkpoint_dir: Path,
    start: Optional[Path]= None,
    device: str = "cuda",
    iters: int = 12,
    search_range: int = 3,
    steps: int=50000,
    lr: float=3e-4,
    image_loss_weight: float=1,
    reg_loss_weight: float=.01,
    log_freq: int=1,
    val_freq: int=1,
    save_freq: int=1,
):
    train_dataset = ImageDataset(data_json, split="train", downsample=stage1_downsample)
    val_dataset = PatchDataset(data_json, split="val", res_factor=stage1_downsample, patch_factor=stage1_downsample)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True) #, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True) #, num_workers=4)

    model = SomeNet(iters=iters, search_range=search_range).to(device)
    step_count = 0
    if start is not None:
        model.load_state_dict(torch.load(start))
        step_count = int(start.name.split("_")[-1].split(".")[0])

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=checkpoint_dir)

    print(f"Dataset size is {len(train_dataset)}")
    print(f"Starting training from step {step_count}")

    while step_count + 1 < steps:
        for step_count, data in zip(trange(step_count, steps), train_loader):
            fixed = data["image"][0,...].unsqueeze(0)
            moving = data["image"][1,...].unsqueeze(0)
            fixed, moving = fixed.to(device), moving.to(device)
            flow, hidden = model(fixed, moving)

            moved = warp_image(flow, moving)

            losses_dict: Dict[str, torch.Tensor] = {}

            losses_dict["image_loss"] = image_loss_weight * MutualInformationLoss()(
                moved.squeeze(), fixed.squeeze()
            )
            losses_dict["grad"] = reg_loss_weight * Grad()(flow)

            if "segmentation" in data:
                fixed_segmentation = data["segmentataion"][0,...].unsqueeze(0)
                moving_segmentation = data["segmentataion"][1,...].unsqueeze(0)

                moved_segmentation = warp_image(flow, moving_mask)

                fixed_segmentation = torch.round(fixed_segmentation)
                moved_segmentation = torch.round(moved_segmentation)

                losses_dict["dice_loss"] = seg_loss_weight * DiceLoss()(
                    fixed_segmenatation, moved_segmentation())

            total_loss = sum(losses_dict.values())
            assert isinstance(total_loss, torch.Tensor)
            losses_dict_log = {k: v.item() for k, v in losses_dict.items()}

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            if step_count % log_freq == 0:
                tb_log(
                    writer,
                    losses_dict_log,
                    step=step_count,
                    moving_fixed_moved=(moving, fixed, moved),
                )

            if val_freq > 0 and step_count % val_freq == 0 and step_count > 0:
                losses_cum_dict = defaultdict(list)
                with torch.no_grad(), evaluating(model):
                    for data in val_loader:
                        fixed, moving = data["fixed_image"], data["moving_image"]
                        fixed, moving = fixed.to(device), moving.to(device)
                        flow, hidden = model(fixed, moving)

                        moved = warp_image(flow, moving)

                        losses_cum_dict["image_loss"].append(
                            (
                                image_loss_weight
                                * MutualInformationLoss()(
                                    moved.squeeze(), fixed.squeeze()
                                )
                            ).item()
                        )
                        losses_cum_dict["grad"].append(
                            (reg_loss_weight * Grad()(flow)).item()
                        )

                        if "fixed_segmentation" in data:
                            fixed_segmentation = data["fixed_segmentation"].to(device).float()
                            moving_segmentation = data["moving_segmentation"].to(device).float()

                            moved_segmentation = warp_image(flow, moving_mask)

                            fixed_segmentation = torch.round(fixed_segmentation)
                            moved_segmentation = torch.round(moved_segmentation)

                            losses_cum_dict["dice_loss"].append(
                                    seg_loss_weight
                                    * DiceLoss()(fixed_segmenatation, moved_segmentation())
                            )


                for k, v in losses_cum_dict.items():
                    writer.add_scalar(
                        f"val_{k}", np.mean(v).item(), global_step=step_count
                    )

            if step_count % save_freq == 0:
                torch.save(
                    model.state_dict(),
                    checkpoint_dir / f"rnn{stage1_downsample}x_{step_count}.pth",
                )

    torch.save(model.state_dict(), checkpoint_dir / f"rnn{stage1_downsample}x_{step_count}.pth")


app1()
