import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import einops
import numpy as np
from typing import Dict, Optional, Union
from torch.utils.tensorboard.writer import SummaryWriter
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from typer import Typer
from tqdm import trange, tqdm

from common import concat_flow, identity_grid_torch, load_keypoints, tb_log, torch2skimage_disp, warp_image
from differentiable_metrics import MINDLoss, Grad, TotalRegistrationLoss
from networks import SomeNet

app = Typer()


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


# FIXME: make this dataset more distinct from stage1 (remove func to not return hidden)
class PatchDatasetStage2(Dataset):
    """
    Extracts 3D `patches' at a specific resolution
    """

    def __init__(
        self,
        data_json: Path,
        res_factor: int,
        artifacts: Path,
        patch_factor: int,
        split: str,
    ):
        super().__init__()
        if res_factor not in [1, 2, 4]:
            raise ValueError(f"Unacceptable resolution factor {res_factor}")

        if patch_factor not in [4]:
            raise ValueError(f"Unacceptable patch factor {patch_factor}")

        with open(data_json, "r") as f:
            data = json.load(f)[split]

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
        self.artifacts = artifacts
        self.check_artifacts()

    def __len__(self):
        return len(self.indexes)

    def check_artifacts(self):
        for index in range(len(self.indexes)):
            data_index, _ = self.indexes[index]
            data = self.data[data_index]

            f, m = "fixed", "moving"
            fname, mname = Path(data[f"{f}_image"]), Path(data[f"{m}_image"])
            flowpath = self.artifacts/(f"flow-{mname.name}2{fname.name}.pt")
            hiddenpath = self.artifacts/(f"hidden-{mname.name}2{fname.name}.pt")
            if not flowpath.exists():
                raise ValueError("Could not find flow ", str(flowpath))
            if not hiddenpath.exists():
                raise ValueError("Could not find hidden ", str(hiddenpath))

    def fold_(self, inp, res_shape, patch_shape): 
        if inp.shape[-1] == 1:
            return inp[...,0]
        else:
            unsqueeze = inp.shape[0] == 1
            inp = inp.squeeze(0)
            inp = einops.rearrange(inp, 'c h d w p -> c (h d w) p')
            r2 = tuple(r//2 for r in res_shape[-2:])
            folded = F.fold(inp, r2, patch_shape[-2:], stride=patch_shape[-2:])
            if unsqueeze: folded = folded.unsqueeze(0)
            return folded

    def load_keypoints_for_patch(self, data, res_shape, patch_shape, patch_index):
        fixed_kps = load_keypoints(data["fixed_keypoints"])/self.res_factor
        moving_kps = load_keypoints(data["moving_keypoints"])/self.res_factor

        grid = identity_grid_torch(res_shape, device="cpu").squeeze()
        grid_patched = F.unfold(grid, patch_shape[-2:], stride=patch_shape[-2:])
        grid_patch = grid_patched[..., patch_index]
        grid_patch = grid_patch.reshape(3, res_shape[-3], *patch_shape[-2:])

        masks = [
            torch.logical_and(
                fixed_kps[:,i] > grid_patch[i,...].min(),
                fixed_kps[:,i] < grid_patch[i,...].max()
            )
            for i in range(3)
        ]
        mask = torch.logical_and(torch.logical_and(masks[0], masks[1]), masks[2])
        if not mask.any():
            return None
        fixed_kps_p = fixed_kps[mask.squeeze(),:]
        moving_kps_p = moving_kps[mask.squeeze(),:]
        min_tensor = torch.stack([grid_patch[i,...].min() for i in range(3)], dim=-1).unsqueeze(0)

        fixed_kps_p = fixed_kps_p - min_tensor
        moving_kps_p = moving_kps_p - min_tensor

        return fixed_kps_p, moving_kps_p


    def __getitem__(self, index: int):
        data_index, patch_index = self.indexes[index]
        data = self.data[data_index]

        r = self.res_factor
        p = self.patch_factor


        f, m = "fixed", "moving"

        fname, mname = Path(data[f"{f}_image"]), Path(data[f"{m}_image"])
        flow = torch.load(self.artifacts/(f"flow-{mname.name}2{fname.name}.pt"), map_location="cpu")
        hidden = torch.load(self.artifacts/(f"hidden-{mname.name}2{fname.name}.pt"), map_location="cpu")

        fixed_nib = nib.load(fname)
        moving_nib = nib.load(mname)

        fixed = torch.from_numpy(fixed_nib.get_fdata()).unsqueeze(0)
        moving = torch.from_numpy(moving_nib.get_fdata()).unsqueeze(0)

        ogshape = fixed_nib.shape[-3:]
        rshape = tuple(i // r for i in fixed.shape[-3:])
        pshape = tuple(i // p for i in ogshape[-3:])

        flow = self.fold_(flow, rshape, pshape)
        hidden = self.fold_(hidden, rshape, pshape)

        fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min())
        moving = (moving - moving.min()) / (moving.max() - moving.min())

        fixed = F.interpolate(fixed.unsqueeze(0), rshape).squeeze(0).float()
        moving = F.interpolate(moving.unsqueeze(0), rshape).squeeze(0).float()

        factor = rshape[-1]//flow.shape[-1]
        flow = F.interpolate(flow, rshape)*factor
        hidden = F.interpolate(hidden, rshape)*factor
        moving = warp_image(flow, moving.unsqueeze(0)).squeeze(0)

        fixed_ps = F.unfold(fixed, pshape[-2:], stride=pshape[-2:])
        moving_ps = F.unfold(moving, pshape[-2:], stride=pshape[-2:])
        hidden_ps = F.unfold(hidden.squeeze(0), pshape[-2:], stride=pshape[-2:])
        flow_ps = F.unfold(flow.squeeze(0), pshape[-2:], stride=pshape[-2:])

        L = fixed_ps.shape[-1]
        assert L == self.n_patches, f"Unexpected number of patches: {L}"

        fixed_p = fixed_ps[..., patch_index]
        moving_p = moving_ps[..., patch_index]
        hidden_p = hidden_ps[..., patch_index]
        flow_p = flow_ps[..., patch_index]

        fixed_p = fixed_p.reshape(1, fixed.shape[-3], *pshape[-2:])
        moving_p = moving_p.reshape(1, fixed.shape[-3], *pshape[-2:])
        hidden_p = hidden_p.reshape(hidden_p.shape[0], fixed.shape[-3], *pshape[-2:])
        flow_p = flow_p.reshape(3, fixed.shape[-3], *pshape[-2:])

        ret: Dict[str, Union[torch.Tensor, int, str]] = {
            "fixed_image": fixed_p,
            "moving_image": moving_p,
            "hidden": hidden_p,
            "flowin": flow_p,
        }
        ret["patch_index"] = patch_index
        ret["fixed_image_name"] = fname.name
        ret["moving_image_name"] = mname.name


        if "fixed_keypoints" in data:
            out = self.load_keypoints_for_patch(data, rshape, pshape, patch_index)
            if out is not None:
                fixed_kps, moving_kps = out
                ret["fixed_keypoints"] = fixed_kps
                ret["moving_keypoints"] = moving_kps
                ret["fixed_spacing"] = torch.Tensor(get_spacing(fixed_nib))
                ret["moving_spacing"] = torch.Tensor(get_spacing(moving_nib))

        return ret


# FIXME: stage1 => patch_size === res_size, so you can simplify the __getitem__
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
    ):
        super().__init__()
        if res_factor not in [1, 2, 4]:
            raise ValueError(f"Unacceptable resolution factor {res_factor}")

        if patch_factor not in [4]:
            raise ValueError(f"Unacceptable patch factor {patch_factor}")

        with open(data_json, "r") as f:
            data = json.load(f)[split]

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

    def load_keypoints_for_patch(self, data, res_shape, patch_shape, patch_index):
        fixed_kps = load_keypoints(data["fixed_keypoints"])/self.res_factor
        moving_kps = load_keypoints(data["moving_keypoints"])/self.res_factor

        grid = identity_grid_torch(res_shape, device="cpu").squeeze()
        grid_patched = F.unfold(grid, patch_shape[-2:], stride=patch_shape[-2:])
        grid_patch = grid_patched[..., patch_index]
        grid_patch = grid_patch.reshape(3, res_shape[-3], *patch_shape[-2:])

        masks = [
            torch.logical_and(
                fixed_kps[:,i] > grid_patch[i,...].min(),
                fixed_kps[:,i] < grid_patch[i,...].max()
            )
            for i in range(3)
        ]
        mask = torch.logical_and(torch.logical_and(masks[0], masks[1]), masks[2])
        if not mask.any():
            return None
        fixed_kps_p = fixed_kps[mask.squeeze(),:]
        moving_kps_p = moving_kps[mask.squeeze(),:]
        min_tensor = torch.stack([grid_patch[i,...].min() for i in range(3)], dim=-1).unsqueeze(0)

        fixed_kps_p = fixed_kps_p - min_tensor
        moving_kps_p = moving_kps_p - min_tensor

        return fixed_kps_p, moving_kps_p

    def __getitem__(self, index: int):
        data_index, patch_index = self.indexes[index]
        data = self.data[data_index]

        r = self.res_factor
        p = self.patch_factor

        f, m = "fixed", "moving"

        fname, mname = Path(data[f"{f}_image"]), Path(data[f"{m}_image"])

        fixed_nib = nib.load(fname)
        moving_nib = nib.load(mname)
        ogshape = fixed_nib.shape[-3:]

        fixed = torch.from_numpy(fixed_nib.get_fdata()).unsqueeze(0)
        moving = torch.from_numpy(moving_nib.get_fdata()).unsqueeze(0)

        fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min())
        moving = (moving - moving.min()) / (moving.max() - moving.min())

        rshape = tuple(i // r for i in fixed.shape[-3:])
        pshape = tuple(i // p for i in ogshape[-3:])
        fixed = F.interpolate(fixed.unsqueeze(0), rshape).squeeze(0).float()
        moving = F.interpolate(moving.unsqueeze(0), rshape).squeeze(0).float()

        if r != self.patch_factor:

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
        ret["fixed_image_name"] = fname.name
        ret["moving_image_name"] = mname.name

        if "fixed_keypoints" in data:
            out = self.load_keypoints_for_patch(data, rshape, pshape, patch_index)
            if out is not None:
                fixed_kps, moving_kps = out
                ret["fixed_keypoints"] = fixed_kps
                ret["moving_keypoints"] = moving_kps
                ret["fixed_spacing"] = torch.Tensor(get_spacing(fixed_nib))
                ret["moving_spacing"] = torch.Tensor(get_spacing(moving_nib))

        return ret

@app.command()
def eval_stage3(
    data_json: Path,
    savedir: Path,
    artifacts: Path,
    res: int,
    checkpoint: Path,
    device: str = "cuda",
    iters: int=4,
    search_range: int=2,
    split: str="val"
):
    """
    Stage3 (Final) eval
    """

    dataset = PatchDatasetStage2(data_json, res, artifacts, 4, split="val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    savedir.mkdir(exist_ok=True)

    model = SomeNet(iters=iters, search_range=search_range).to(device)
    model.load_state_dict(torch.load(checkpoint))

    with torch.no_grad(), evaluating(model):
        flows = defaultdict(list)
        for data in tqdm(loader):
            fixed, moving = data["fixed_image"], data["moving_image"]
            fixed, moving = fixed.to(device), moving.to(device)
            flow, _ = model(fixed, moving)
            savename = f'{data["moving_image_name"][0]}2{data["fixed_image_name"][0]}.pt'
            flows[savename].append((data["patch_index"], flow.detach().cpu()))

            flow, _ = model(moving, fixed)
            flowin = data["flowin"].to(device)
            flow = concat_flow(flowin, flow)

            savename = f'{data["fixed_image_name"][0]}2{data["moving_image_name"][0]}.pt'
            flows[savename].append((data["patch_index"], flow.detach().cpu()))

        for k, v in flows.items():
            try:
                fk = torch.stack([i[1] for i in sorted(v)], dim=-1)
                pshape = fk.shape[-3:-1]
                res_shape = (fk.shape[2], *[i*4 for i in pshape])
                inp = fk.squeeze(0)
                inp = einops.rearrange(inp, 'c h d w p -> c (h d w) p')
                folded = F.fold(inp, res_shape[-2:], pshape, stride=pshape)
                folded = folded.unsqueeze(0)
                moving, fixed = k.split('z2')
                moving = moving + "z"
                fixed = ".".join(fixed.split(".")[:-1]) # remove .pt
                disp_name = f"disp_{fixed[-16:-12]}_{moving[-16:-12]}"
                disp_np = torch2skimage_disp(folded)
                disp_nib = nib.Nifti1Image(disp_np, affine=np.eye(4))
                nib.save(disp_nib, savedir/f"{disp_name}.nii.gz")
            except:
                print(k)
                continue


@app.command()
def eval_stage2(
    data_json: Path,
    savedir: Path,
    artifacts: Path,
    res: int,
    checkpoint: Path,
    device: str = "cuda",
    iters: int=8,
    search_range: int=3,
    split="val"
):
    """
    Stage2 eval
    """

    dataset = PatchDatasetStage2(data_json, res, artifacts, 4, split=split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    savedir.mkdir(exist_ok=True)

    model = SomeNet(iters=iters, search_range=search_range).to(device)
    model.load_state_dict(torch.load(checkpoint))

    with torch.no_grad(), evaluating(model):
        for i in trange(0, len(dataset), dataset.n_patches):
            flows, hiddens = defaultdict(list), defaultdict(list)
            for j in range(i, i+dataset.n_patches):
                data = dataset[j]
                fixed, moving = data["fixed_image"], data["moving_image"]
                assert isinstance(fixed, torch.Tensor)
                assert isinstance(moving, torch.Tensor)
                fixed, moving = fixed.unsqueeze(0).to(device), moving.unsqueeze(0).to(device)
                flow, hidden = model(fixed, moving)
                savename = f'{data["moving_image_name"]}2{data["fixed_image_name"]}.pt'
                flowin = data["flowin"]
                assert isinstance(flowin, torch.Tensor)
                flow = concat_flow(flowin.to(device), flow)

                flows[savename].append((data["patch_index"], flow.detach().cpu()))
                hiddens[savename].append((data["patch_index"], hidden.detach().cpu()))

            for k, v in flows.items():

                fk = torch.stack([i[1] for i in sorted(v, key=lambda x: x[0])], dim=-1)
                hk = torch.stack([i[1] for i in sorted(hiddens[k], key=lambda x: x[0])], dim=-1)
                torch.save(fk, savedir/("flow-" + k))
                torch.save(hk, savedir/("hidden-" + k))



@app.command()
def eval_stage1(
    data_json: Path,
    savedir: Path,
    res: int,
    checkpoint: Path,
    device: str = "cuda",
    iters: int=12,
    search_range: int=3,
    split="val"
):
    """
    Stage1 training
    """

    dataset = PatchDataset(data_json, res, 4, split=split)

    savedir.mkdir(exist_ok=True)

    model = SomeNet(iters=iters, search_range=search_range).to(device)
    model.load_state_dict(torch.load(checkpoint))

    with torch.no_grad(), evaluating(model):
        for i in trange(0, len(dataset), dataset.n_patches):
            flows, hiddens = defaultdict(list), defaultdict(list)
            for j in range(i, i+dataset.n_patches):
                data = dataset[j]
                fixed, moving = data["fixed_image"], data["moving_image"]
                assert isinstance(fixed, torch.Tensor)
                assert isinstance(moving, torch.Tensor)
                fixed, moving = fixed.unsqueeze(0).to(device), moving.unsqueeze(0).to(device)
                flow, hidden = model(fixed, moving)
                savename = f'{data["moving_image_name"]}2{data["fixed_image_name"]}.pt'
                flows[savename].append((data["patch_index"], flow.detach().cpu()))
                hiddens[savename].append((data["patch_index"], hidden.detach().cpu()))

            assert len(flows.keys()) == 1, "Expected only one key"
            for k, v in flows.items():
                fk = torch.stack([i[1] for i in sorted(v, key=lambda x: x[0])], dim=-1)
                hk = torch.stack([i[1] for i in sorted(hiddens[k], key=lambda x: x[0])], dim=-1)
                torch.save(fk, savedir/("flow-" + k))
                torch.save(hk, savedir/("hidden-" + k))

            del flows, hiddens



@app.command()
def train_stage2(
    data_json: Path,
    checkpoint_dir: Path,
    artifacts: Path,
    res: int,
    start: Optional[Path] = None,
    steps: int = 10000,
    lr: float = 3e-4,
    device: str = "cuda",
    image_loss_weight: float = 1,
    reg_loss_weight: float = 0.01,
    kp_loss_weight: float=1,
    log_freq: int = 100,
    save_freq: int = 100,
    val_freq: int = 1000,
    iters: int=4,
    search_range: int=1
):
    """
    Stage2 training
    """

    train_dataset = PatchDatasetStage2(data_json, res, artifacts, 4, split="train")
    val_dataset = PatchDatasetStage2(data_json, res, artifacts,4, split="val")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    checkpoint_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=checkpoint_dir)

    model = SomeNet(search_range=search_range, iters=iters).to(device)
    step_count = 0
    if start is not None:
        model.load_state_dict(torch.load(start))
        step_count = int(start.name.split("_")[-1].split(".")[0])

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training from step {step_count}")
    while step_count < steps:
        for step_count, data in zip(trange(step_count, steps), train_loader):
            fixed, moving, hidden = data["fixed_image"], data["moving_image"], data["hidden"]
            fixed, moving, hidden = fixed.to(device), moving.to(device), hidden.to(device)
            flow, hidden = model(fixed, moving, hidden)

            moved = warp_image(flow, moving)

            losses_dict: Dict[str, torch.Tensor] = {}
            losses_dict["image_loss"] = image_loss_weight * MINDLoss()(
                moved.squeeze(), fixed.squeeze()
            )
            losses_dict["grad"] = reg_loss_weight * Grad()(flow)

            if "fixed_keypoints" in data:
                flowin = data["flowin"].to(device)
                flow = concat_flow(flowin, flow)
                losses_dict["keypoints"] = kp_loss_weight * TotalRegistrationLoss()(
                    fixed_landmarks=data["fixed_keypoints"].squeeze(0),
                    moving_landmarks=data["moving_keypoints"].squeeze(0),
                    displacement_field=flow,
                    fixed_spacing=data["fixed_spacing"].squeeze(0),
                    moving_spacing=data["moving_spacing"].squeeze(0),
                )

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

            if val_freq > 0 and step_count % val_freq == 0 and step_count>0:
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

                        if "fixed_keypoints" in data:
                            flowin = data["flowin"].to(device)
                            flow = concat_flow(flowin, flow)
                            losses_cum_dict["keypoints"].append(kp_loss_weight * TotalRegistrationLoss()(
                                fixed_landmarks=data["fixed_keypoints"].squeeze(0),
                                moving_landmarks=data["moving_keypoints"].squeeze(0),
                                displacement_field=flow,
                                fixed_spacing=data["fixed_spacing"].squeeze(0),
                                moving_spacing=data["moving_spacing"].squeeze(0),
                            ).item())

                for k, v in losses_cum_dict.items():
                    writer.add_scalar(
                        f"val_{k}", np.mean(v).item(), global_step=step_count
                    )

            if step_count % save_freq == 0:
                torch.save(
                    model.state_dict(), checkpoint_dir / f"rnn{res}x_{step_count}.pth",
                )

    torch.save(model.state_dict(), checkpoint_dir / f"rnn{res}x_{step_count}.pth")


@app.command()
def train_stage1(
    data_json: Path,
    checkpoint_dir: Path,
    res: int,
    start: Optional[Path] = None,
    steps: int = 10000,
    lr: float = 1e-4,
    device: str = "cuda",
    image_loss_weight: float = 1,
    reg_loss_weight: float = 0.1,
    kp_loss_weight: float=1,
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

            if "fixed_keypoints" in data:
                losses_dict["keypoints"] = kp_loss_weight * TotalRegistrationLoss()(
                    fixed_landmarks=data["fixed_keypoints"].squeeze(0),
                    moving_landmarks=data["moving_keypoints"].squeeze(0),
                    displacement_field=flow,
                    fixed_spacing=data["fixed_spacing"].squeeze(0),
                    moving_spacing=data["moving_spacing"].squeeze(0),
                )

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

                        if "fixed_keypoints" in data:
                            losses_cum_dict["keypoints"].append(kp_loss_weight * TotalRegistrationLoss()(
                                fixed_landmarks=data["fixed_keypoints"].squeeze(0),
                                moving_landmarks=data["moving_keypoints"].squeeze(0),
                                displacement_field=flow,
                                fixed_spacing=data["fixed_spacing"].squeeze(0),
                                moving_spacing=data["moving_spacing"].squeeze(0),
                            ).item())


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
