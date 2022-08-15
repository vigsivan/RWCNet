import json
import random
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import einops
import numpy as np
from typing import Dict, List, Optional, Union
from torch.utils.tensorboard.writer import SummaryWriter
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from typer import Typer
from tqdm import trange
from dataclasses import dataclass

from common import warp_image, tb_log
from differentiable_metrics import MINDLoss, Grad
from networks import SomeNet

app = Typer()


@dataclass
class PatchData:
    fixed_image: torch.Tensor
    moving_image: torch.Tensor
    fixed_image_name: str
    moving_image_name: str
    patch_index: int


add_bc_dim = lambda x: einops.repeat(x, "d h w -> b c d h w", b=1, c=1).float()
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


class PatchDataset(Dataset):
    """
    Extracts 3D `patches' at a specific resolution
    """

    def __init__(
        self,
        data_json: Path,
        res_factor: int,
        patch_factor: int,
        split: str,
        random_switch: bool = True,
    ):
        super().__init__()
        if res_factor not in [1, 2, 4]:
            raise ValueError(f"Unacceptable resolution factor {res_factor}")

        if patch_factor not in [4]:
            raise ValueError(f"Unacceptable patch factor {patch_factor}")

        with open(data_json, "r") as f:
            data = json.load(f)[split]

        self.random_switch = random_switch

        # FIXME
        # I'm sure the heuristic here is like 2**(patch_factor/res_factor) or
        # something but I'm too lazy to figure it out
        if res_factor == 4:
            n_patches = 1
        elif res_factor == 2:
            n_patches = 4
        else:
            n_patches = 16

        self.data = data
        self.indexes = [(i, j) for i, _ in enumerate(data) for j in range(n_patches)]
        self.res_factor = res_factor
        self.patch_factor = patch_factor
        self.n_patches = n_patches

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index: int):
        data_index, patch_index = self.indexes[index]
        data = self.data[data_index]

        r = self.res_factor
        p = self.patch_factor

        f, m = "fixed", "moving"
        if self.random_switch and random.randint(0, 10) % 2 == 0:
            f, m = m, f

        fname, mname = data[f"{f}_image"], data[f"{m}_image"]

        fixed_nib = nib.load(fname)
        moving_nib = nib.load(mname)
        ogshape = fixed_nib.shape[-3:]

        fixed = torch.from_numpy(fixed_nib.get_fdata()).unsqueeze(0)
        moving = torch.from_numpy(moving_nib.get_fdata()).unsqueeze(0)

        fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min())
        moving = (moving - moving.min()) / (moving.max() - moving.min())

        rshape = tuple(i // r for i in fixed.shape[-3:])
        fixed = F.interpolate(fixed.unsqueeze(0), rshape).squeeze(0).float()
        moving = F.interpolate(moving.unsqueeze(0), rshape).squeeze(0).float()

        if r != self.patch_factor:
            pshape = tuple(i // p for i in ogshape[-3:])

            fixed_ps = F.unfold(fixed, pshape[-2:], stride=pshape[-2:])
            moving_ps = F.unfold(moving, pshape[-2:], stride=pshape[-2:])

            L = fixed_ps.shape[-1]
            assert L == self.n_patches, f"Unexpected number of patches: {L}"

            fixed_p = fixed_ps[..., patch_index]
            moving_p = moving_ps[..., patch_index]

            fixed_p = fixed_p.reshape(1, fixed.shape[-3], *pshape[-2:])
            moving_p = moving_p.reshape(1, fixed.shape[-3], *pshape[-2:])

            ret: Dict[str, Union[torch.Tensor, int, str]] = {
                "fixed_image": fixed_p,
                "moving_image": moving_p,
            }
        else:
            ret = {
                "fixed_image": fixed,
                "moving_image": moving,

            }

        ret["patch_index"] = patch_index
        ret["fixed_image_name"] = fname
        ret["moving_image_name"] = mname

        return ret

@app.command()
def eval_stage1(
    data_json: Path,
    savedir: Path,
    res: int,
    checkpoint: Path,
    steps: int = 10000,
    device: str = "cuda",
):
    """
    Stage1 training
    """

    train_dataset = PatchDataset(data_json, res, 4, split="train")
    val_dataset = PatchDataset(data_json, res, 4, split="val")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    savedir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=savedir)

    model = SomeNet().to(device)
    model.load_state_dict(torch.load(checkpoint))

    patches = train_dataset.n_patches

    with torch.no_grad(), evaluating(model):
        flows, hiddens = defaultdict(list), defaultdict(list)
        for loader in (train_loader, val_loader):
            for data in loader:
                fixed, moving = data["fixed_image"], data["moving_image"]
                fixed, moving = fixed.to(device), moving.to(device)
                flow, hidden = model(fixed, moving)
                savename = f'{data["moving_image_name"]}2{data["fixed_image_name"]}.pt'
                flows[savename].append((data.patch_index, flow))
                hiddens[savename].append((data.patch_index, hidden))

                flow, hidden = model(moving, fixed)
                savename = f'{data["moving_image_name"]}2{data["fixed_image_name"]}.pt'
                flows[savename].append((data.patch_index, flow))
                hiddens[savename].append((data.patch_index, hidden))

        breakpoint()
        flows = torch.stack(sorted(flows), dim=-1)
        hiddens = torch.stack(sorted(hiddens), dim=-1)



@app.command()
def train_stage1(
    data_json: Path,
    checkpoint_dir: Path,
    res: int,
    start: Optional[Path] = None,
    steps: int = 10000,
    lr: float = 1e-4,
    train_paired: bool = True,
    val_paired: bool = True,
    device: str = "cuda",
    image_loss_weight: float = 1,
    dice_loss_weight: float = 10.0,
    reg_loss_weight: float = 0.1,
    kp_loss_weight: float = 1,
    log_freq: int = 5,
    save_freq: int = 50,
    val_freq: int = 50,
):
    """
    Stage1 training
    """

    train_dataset = PatchDataset(data_json, res, 4, split="train")
    val_dataset = PatchDataset(data_json, res, 4, split="val")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    checkpoint_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=checkpoint_dir)

    model = SomeNet().to(device)
    step_count = 0
    if start is not None:
        model.load_state_dict(torch.load(start))
        if "step" in start.name:
            step_count = int(start.name.split("_step")[-1].split(".")[0])

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training from step {step_count}")
    while step_count < steps:
        for step_count, data in zip(trange(step_count, steps), train_loader):

            fixed, moving = data["fixed_image"], data["moving_image"]
            fixed, moving = fixed.to(device), moving.to(device)
            flow, hidden = model(fixed, moving)

            moved = warp_image(flow, moving)

            losses_dict: Dict[str, torch.Tensor] = {}
            losses_dict["image_loss"] = image_loss_weight * MINDLoss()(
                moved.squeeze(), fixed.squeeze()
            )
            losses_dict["grad"] = reg_loss_weight * Grad()(flow)

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

            if val_freq > 0 and step_count % val_freq == 0:
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
                                * MINDLoss()(moved.squeeze(), fixed.squeeze())
                            ).item()
                        )
                        losses_cum_dict["grad"].append(
                            (reg_loss_weight * Grad()(flow)).item()
                        )

                for k, v in losses_cum_dict.items():
                    writer.add_scalar(
                        f"val_{k}", np.mean(v).item(), global_step=step_count
                    )

            if step_count % save_freq == 0:
                torch.save(
                    model.state_dict(), checkpoint_dir / f"rnn{res}x_{step_count}.pth",
                )

    torch.save(model.state_dict(), checkpoint_dir / f"rnn{res}x_{step_count}.pth")


app()
