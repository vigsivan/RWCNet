import random
import json
from collections import defaultdict
from functools import partial
from contextlib import contextmanager, nullcontext
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
from data import InfiniteDataLoader
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
    MSE,
    DiceLoss,
    Grad,
    TotalRegistrationLoss,
    MutualInformationLoss,
    NCC,
)
from networks import SomeNet, SomeNetNoCorr, SomeNetNoisy

__all__ = ["train", "train_with_artifacts"]

app = Typer()

get_spacing = lambda x: np.sqrt(np.sum(x.affine[:3, :3] * x.affine[:3, :3], axis=0))


def get_loss_fn(loss_fn: str) -> Union[MutualInformationLoss, NCC, MSE]:
    if loss_fn == "mi":
        return MutualInformationLoss()
    elif loss_fn == "ncc":
        return NCC()
    else:
        return MSE()


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


class PatchDatasetWithArtifacts(Dataset):
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
        dset_min: float,
        dset_max: float,
    ):
        super().__init__()
        if res_factor not in [1, 2, 4]:
            raise ValueError(f"Unacceptable resolution factor {res_factor}")

        if patch_factor not in [1, 2, 4]:
            raise ValueError(f"Unacceptable patch factor {patch_factor}")

        with open(data_json, "r") as f:
            data = json.load(f)[split]

        r, p = 1 / res_factor, 1 / patch_factor
        n_patches = (((r - p) / p) + 1) ** 2

        if n_patches.is_integer():
            n_patches = int(n_patches)
        else:
            raise Exception(f"Number of patches is not integer, is {n_patches}")

        chan_split = patch_factor // res_factor

        self.normalize = lambda x: (x - dset_min) / (dset_max - dset_min)

        self.data = data
        self.indexes = [
            (i, j, k)
            for i, _ in enumerate(data)
            for j in range(chan_split)
            for k in range(n_patches)
        ]
        self.res_factor = res_factor
        self.patch_factor = patch_factor
        self.n_patches = n_patches
        self.artifacts = artifacts
        self.chan_split = chan_split
        self.check_artifacts()
        self.split = split

    def __len__(self):
        return len(self.indexes)

    def check_artifacts(self):
        for index in range(len(self.indexes)):
            data_index, _, _ = self.indexes[index]
            data = self.data[data_index]

            f, m = "fixed", "moving"
            fname, mname = Path(data[f"{f}_image"]), Path(data[f"{m}_image"])
            flowpath = self.artifacts / (f"flow-{mname.name}2{fname.name}.pt")
            hiddenpath = self.artifacts / (f"hidden-{mname.name}2{fname.name}.pt")
            if not flowpath.exists():
                raise ValueError("Could not find flow ", str(flowpath))
            if not hiddenpath.exists():
                raise ValueError("Could not find hidden ", str(hiddenpath))

    def fold_(self, inp, res_shape, patch_shape):
        if inp.shape[-1] == 1:
            return inp[..., 0]
        else:
            unsqueeze = inp.shape[0] == 1
            inp = inp.squeeze(0)
            inp = einops.rearrange(inp, "c h d w p -> c (h d w) p")
            r2 = tuple(r // 2 for r in res_shape[-2:])
            folded = F.fold(inp, r2, patch_shape[-2:], stride=patch_shape[-2:])
            if unsqueeze:
                folded = folded.unsqueeze(0)
            return folded

    def load_keypoints_for_patch(
        self, data, res_shape, patch_shape, chan_index, patch_index
    ):
        fixed_kps = load_keypoints(data["fixed_keypoints"]) / self.res_factor
        moving_kps = load_keypoints(data["moving_keypoints"]) / self.res_factor

        grid = identity_grid_torch(res_shape, device="cpu").squeeze()
        grid_patched = F.unfold(grid, patch_shape[-2:], stride=patch_shape[-2:])
        grid_patch = grid_patched[..., patch_index]
        grid_patch = grid_patch.reshape(3, res_shape[-3], *patch_shape[-2:])
        grid_patch = torch.split(
            grid_patch, [res_shape[-3] // self.chan_split] * self.chan_split, dim=1
        )[chan_index]

        masks = [
            torch.logical_and(
                fixed_kps[:, i] > grid_patch[i, ...].min(),
                fixed_kps[:, i] < grid_patch[i, ...].max(),
            )
            for i in range(3)
        ]
        mask = torch.logical_and(torch.logical_and(masks[0], masks[1]), masks[2])
        if not mask.any():
            return None
        fixed_kps_p = fixed_kps[mask.squeeze(), :]
        moving_kps_p = moving_kps[mask.squeeze(), :]
        min_tensor = torch.stack(
            [grid_patch[i, ...].min() for i in range(3)], dim=-1
        ).unsqueeze(0)

        fixed_kps_p = fixed_kps_p - min_tensor
        moving_kps_p = moving_kps_p - min_tensor

        return fixed_kps_p, moving_kps_p

    def get_patch(
        self,
        tensor: torch.Tensor,
        n_channels: int,
        rshape: Tuple[int, int, int],
        pshape: Tuple[int, int, int],
        patch_index: int,
        chan_index: int,
    ):

        assert len(tensor.shape) == 4, "Expected tensor to have four dimensions"
        tensor_ps = F.unfold(tensor, pshape[-2:], stride=pshape[-2:])

        L = tensor_ps.shape[-1]
        assert L == self.n_patches, f"Unexpected number of patches: {L}"

        tensor_p = tensor_ps[..., patch_index]
        tensor_p = tensor_p.reshape(n_channels, tensor.shape[-3], *pshape[-2:])
        tensor_p = torch.split(
            tensor_p, [tensor.shape[-3] // self.chan_split] * self.chan_split, dim=1
        )[chan_index]

        return tensor_p

    def load_data_(self, data_index):

        r = self.res_factor
        p = self.patch_factor

        f, m = "fixed", "moving"

        data = self.data[data_index]
        fname, mname = Path(data[f"{f}_image"]), Path(data[f"{m}_image"])
        flow = torch.load(
            self.artifacts / (f"flow-{mname.name}2{fname.name}.pt"), map_location="cpu"
        )
        hidden = torch.load(
            self.artifacts / (f"hidden-{mname.name}2{fname.name}.pt"),
            map_location="cpu",
        )

        fixed_nib = nib.load(fname)
        moving_nib = nib.load(mname)

        fixed = torch.from_numpy(fixed_nib.get_fdata()).unsqueeze(0)
        moving = torch.from_numpy(moving_nib.get_fdata()).unsqueeze(0)

        ogshape = fixed_nib.shape[-3:]
        rshape = tuple(i // r for i in fixed.shape[-3:])
        pshape = tuple(i // p for i in ogshape[-3:])

        fixed = self.normalize(fixed)
        moving = self.normalize(moving)

        fixed = F.interpolate(fixed.unsqueeze(0), rshape).squeeze(0).float()
        moving = F.interpolate(moving.unsqueeze(0), rshape).squeeze(0).float()

        factor = rshape[-1] // flow.shape[-1]

        flow = flow.squeeze().unsqueeze(0).float()
        hidden = hidden.squeeze().unsqueeze(0)

        flow = F.interpolate(flow, rshape) * factor
        hidden = F.interpolate(hidden, rshape) * factor

        moving_i = moving
        moving = warp_image(flow, moving.unsqueeze(0)).squeeze(0)

        hidden = hidden.squeeze(0)
        flow = flow.squeeze(0)

        ret: Dict[str, Union[torch.Tensor, str, Tuple]] = {
            "fixed_image": fixed,
            "moving_image": moving,
            "hidden": hidden,
            "flow": flow,
            "rshape": rshape,
            "pshape": pshape,
            "fname": fname,
            "mname": mname,
        }

        if "fixed_segmentation" in data:

            fixed_seg_nib = nib.load(data["fixed_segmentation"])
            moving_seg_nib = nib.load(data["moving_segmentation"])

            fixed_seg = torch.from_numpy(fixed_seg_nib.get_fdata()).unsqueeze(0)
            moving_seg = torch.from_numpy(moving_seg_nib.get_fdata()).unsqueeze(0)

            fixed_seg = (
                F.interpolate(fixed_seg.unsqueeze(0), rshape, mode="nearest")
                .squeeze(0)
                .float()
            )
            moving_seg = (
                F.interpolate(moving_seg.unsqueeze(0), rshape, mode="nearest")
                .squeeze(0)
                .float()
            )

            moving_seg = warp_image(
                flow.unsqueeze(0), moving_seg.unsqueeze(0), mode="nearest"
            ).squeeze(0)

            ret["fixed_segmentation"] = fixed_seg
            ret["moving_segmentation"] = moving_seg

        if "fixed_mask" in data:

            fixed_mask_nib = nib.load(data["fixed_mask"])
            moving_mask_nib = nib.load(data["moving_mask"])

            fixed_mask = torch.from_numpy(fixed_mask_nib.get_fdata()).unsqueeze(0)
            moving_mask = torch.from_numpy(moving_mask_nib.get_fdata()).unsqueeze(0)

            fixed_mask = (
                F.interpolate(fixed_mask.unsqueeze(0), rshape, mode="nearest")
                .squeeze(0)
                .float()
            )
            moving_mask = (
                F.interpolate(moving_mask.unsqueeze(0), rshape, mode="nearest")
                .squeeze(0)
                .float()
            )

            moving_mask = warp_image(
                flow.unsqueeze(0), moving_mask.unsqueeze(0)
            ).squeeze(0)

            ret["fixed_masked"] = fixed_mask * fixed
            ret["moving_masked"] = warp_image(
                flow.unsqueeze(0), (moving_mask * moving_i).unsqueeze(0)
            ).squeeze(0)

            ret["fixed_mask"] = fixed_mask
            ret["moving_mask"] = moving_mask

        if "fixed_keypoints" in data:
            ret["fixed_keypoints"] = data["fixed_keypoints"]
            ret["moving_keypoints"] = data["moving_keypoints"]
            ret["fixed_spacing"] = torch.Tensor(get_spacing(fixed_nib))
            ret["moving_spacing"] = torch.Tensor(get_spacing(moving_nib))

        return ret

    def get_patch_data(self, data: Dict, chan_index: int, patch_index: int):
        (fixed, moving, flow, hidden, rshape, pshape, fname, mname) = (
            data["fixed_image"],
            data["moving_image"],
            data["flow"],
            data["hidden"],
            data["rshape"],
            data["pshape"],
            data["fname"],
            data["mname"],
        )

        fixed_p = self.get_patch(fixed, 1, rshape, pshape, patch_index, chan_index)
        moving_p = self.get_patch(moving, 1, rshape, pshape, patch_index, chan_index)
        hidden_p = self.get_patch(
            hidden, hidden.shape[0], rshape, pshape, patch_index, chan_index
        )
        flow_p = self.get_patch(flow, 3, rshape, pshape, patch_index, chan_index)

        ret: Dict[str, Union[torch.Tensor, int, str]] = {
            "fixed_image": fixed_p,
            "moving_image": moving_p,
            "hidden": hidden_p,
            "flowin": flow_p,
        }

        ret["patch_index"] = patch_index
        ret["chan_index"] = chan_index
        ret["fixed_image_name"] = fname.name
        ret["moving_image_name"] = mname.name

        if "fixed_keypoints" in data:
            out = self.load_keypoints_for_patch(
                data, rshape, pshape, chan_index, patch_index
            )
            if out is not None:
                fixed_kps, moving_kps = out
                ret["fixed_keypoints"] = fixed_kps
                ret["moving_keypoints"] = moving_kps
                ret["fixed_spacing"] = data["fixed_spacing"]
                ret["moving_spacing"] = data["moving_spacing"]

        if "fixed_segmentation" in data:

            fixed_seg, moving_seg = (
                data["fixed_segmentation"],
                data["moving_segmentation"],
            )

            fixed_seg_p = self.get_patch(
                fixed_seg, 1, rshape, pshape, patch_index, chan_index
            )
            moving_seg_p = self.get_patch(
                moving_seg, 1, rshape, pshape, patch_index, chan_index
            )

            ret["fixed_segmentation"] = fixed_seg_p.long()
            ret["moving_segmentation"] = moving_seg_p.long()

        if "fixed_mask" in data:

            fixed_mask, moving_mask = data["fixed_mask"], data["moving_mask"]
            fixed_masked, moving_masked = data["fixed_masked"], data["moving_masked"]

            fixed_mask_p = self.get_patch(
                fixed_mask, 1, rshape, pshape, patch_index, chan_index
            )
            moving_mask_p = self.get_patch(
                moving_mask, 1, rshape, pshape, patch_index, chan_index
            )

            fixed_masked_p = self.get_patch(
                fixed_masked, 1, rshape, pshape, patch_index, chan_index
            )
            moving_masked_p = self.get_patch(
                moving_masked, 1, rshape, pshape, patch_index, chan_index
            )

            ret["fixed_mask"] = fixed_mask_p.long()
            ret["moving_mask"] = moving_mask_p.long()
            ret["fixed_masked"] = fixed_masked_p
            ret["moving_masked"] = moving_masked_p

        return ret

    def __getitem__(self, index: int):
        data_index, chan_index, patch_index = self.indexes[index]
        data = self.load_data_(data_index)

        patch_data = self.get_patch_data(data, chan_index, patch_index)

        return patch_data


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
        dset_min: float,
        dset_max: float,
        switch: bool = False,
    ):
        super().__init__()

        assert (
            patch_factor == res_factor
        ), "Stage 1 training assumes patch factor equals res factor"
        if res_factor not in [1, 2, 4]:
            raise ValueError(f"Unacceptable resolution factor {res_factor}")

        with open(data_json, "r") as f:
            data = json.load(f)[split]

        self.normalize = lambda x: (x - dset_min) / (dset_max - dset_min)

        self.data = data
        self.indexes = [(i, 0) for i, _ in enumerate(data)]
        self.res_factor = res_factor
        self.patch_factor = patch_factor
        self.n_patches = 1
        self.switch = switch

    def __len__(self):
        return len(self.indexes)

    def load_keypoints_for_patch(self, data, res_shape, patch_shape, patch_index):
        fixed_kps = load_keypoints(data["fixed_keypoints"]) / self.res_factor
        moving_kps = load_keypoints(data["moving_keypoints"]) / self.res_factor

        grid = identity_grid_torch(res_shape, device="cpu").squeeze()
        grid_patched = F.unfold(grid, patch_shape[-2:], stride=patch_shape[-2:])
        grid_patch = grid_patched[..., patch_index]
        grid_patch = grid_patch.reshape(3, res_shape[-3], *patch_shape[-2:])

        masks = [
            torch.logical_and(
                fixed_kps[:, i] > grid_patch[i, ...].min(),
                fixed_kps[:, i] < grid_patch[i, ...].max(),
            )
            for i in range(3)
        ]
        mask = torch.logical_and(torch.logical_and(masks[0], masks[1]), masks[2])
        if not mask.any():
            return None
        fixed_kps_p = fixed_kps[mask.squeeze(), :]
        moving_kps_p = moving_kps[mask.squeeze(), :]
        min_tensor = torch.stack(
            [grid_patch[i, ...].min() for i in range(3)], dim=-1
        ).unsqueeze(0)

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

        fixed = self.normalize(fixed)
        moving = self.normalize(moving)

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

        if "fixed_segmentation" in data:
            fname, mname = Path(data[f"{f}_segmentation"]), Path(
                data[f"{m}_segmentation"]
            )

            fixed_nib = nib.load(fname)
            moving_nib = nib.load(mname)

            fixed_seg = torch.from_numpy(fixed_nib.get_fdata()).unsqueeze(0)
            moving_seg = torch.from_numpy(moving_nib.get_fdata()).unsqueeze(0)

            fixed_seg = F.interpolate(fixed_seg.unsqueeze(0), rshape).squeeze(0).float()
            moving_seg = (
                F.interpolate(moving_seg.unsqueeze(0), rshape).squeeze(0).float()
            )

            ret["fixed_segmentation"] = fixed_seg
            ret["moving_segmentation"] = moving_seg

        if "fixed_mask" in data:
            fname, mname = Path(data[f"{f}_mask"]), Path(data[f"{m}_mask"])

            fixed_nib = nib.load(fname)
            moving_nib = nib.load(mname)

            fixed_mask = torch.from_numpy(fixed_nib.get_fdata()).unsqueeze(0)
            moving_mask = torch.from_numpy(moving_nib.get_fdata()).unsqueeze(0)

            fixed_mask = (
                F.interpolate(fixed_mask.unsqueeze(0), rshape).squeeze(0).float()
            )
            moving_mask = (
                F.interpolate(moving_mask.unsqueeze(0), rshape).squeeze(0).float()
            )

            ret["fixed_mask"] = fixed_mask
            ret["moving_mask"] = moving_mask

            ret["fixed_masked"] = fixed_mask * fixed
            ret["moving_masked"] = moving_mask * moving

        if self.switch and random.choice([0, 1]) == 0:
            ret_switched = {}
            for k, v in ret.items():
                if "fixed" in k:
                    new_k = k.replace("fixed", "moving")
                elif "moving" in k:
                    new_k = k.replace("moving", "fixed")
                else:
                    new_k = k
                ret_switched[new_k] = v

            ret = ret_switched

        return ret


@app.command()
def eval_stage3(
    data_json: Path,
    savedir: Path,
    artifacts: Path,
    res: int,
    dset_min: float,
    dset_max: float,
    checkpoint: Path,
    device: str = "cuda",
    iters: int = 8,
    search_range: int = 3,
    patch_factor: int = 4,
    split="val",
    diffeomorphic: bool = True,
):
    """
    Stage3 eval
    """

    with open(data_json, "r") as f:
        if split not in json.load(f):
            return

    dataset = PatchDatasetWithArtifacts(
        data_json,
        res,
        artifacts,
        patch_factor,
        split=split,
        dset_min=dset_min,
        dset_max=dset_max,
    )

    savedir.mkdir(exist_ok=True)

    model = SomeNet(
        iters=iters, search_range=search_range, diffeomorphic=diffeomorphic
    ).to(device)
    model.load_state_dict(torch.load(checkpoint))

    with torch.no_grad(), evaluating(model):
        for i in trange(0, len(dataset), dataset.n_patches * dataset.chan_split):
            flows = defaultdict(list)
            for j in range(i, i + (dataset.n_patches * dataset.chan_split)):
                data = dataset[j]
                fixed, moving = data["fixed_image"], data["moving_image"]
                assert isinstance(fixed, torch.Tensor)
                assert isinstance(moving, torch.Tensor)
                fixed, moving = fixed.unsqueeze(0).to(device), moving.unsqueeze(0).to(
                    device
                )
                flow, _ = model(fixed, moving)

                savename = f'{data["moving_image_name"]}2{data["fixed_image_name"]}.pt'
                flowin = data["flowin"]
                assert isinstance(flowin, torch.Tensor)
                flowin = flowin.squeeze().unsqueeze(0)
                flow = concat_flow(flow, flowin.to(device))

                flows[savename].append(
                    (data["chan_index"], data["patch_index"], flow.detach().cpu())
                )

            assert len(list(flows.keys())) == 1
            for k, v in flows.items():
                fchannels = defaultdict(list)

                for i in v:
                    fchannels[i[0]].append((i[1], i[2]))

                ffchannels = [
                    torch.stack(
                        [i[1] for i in sorted(fchan, key=lambda x: x[0])], dim=-1
                    )
                    for fchan in fchannels.values()
                ]
                fk = torch.cat(ffchannels, dim=2)

                pshape = fk.shape[-3:-1]
                res_shape = (fk.shape[2], *[i * 2 for i in pshape])

                fk = fk.squeeze(0)
                fk = einops.rearrange(fk, "c h d w p -> c (h d w) p")
                folded_flow = F.fold(fk, res_shape[-2:], pshape, stride=pshape)

                moving, fixed = k.split("z2")
                moving = moving + "z"
                fixed = ".".join(fixed.split(".")[:-1])  # remove .pt
                disp_name = f"disp_{fixed[-16:-12]}_{moving[-16:-12]}"
                disp_np = torch2skimage_disp(folded_flow.unsqueeze(0))
                disp_nib = nib.Nifti1Image(disp_np, affine=np.eye(4))

                nib.save(disp_nib, savedir / f"{disp_name}.nii.gz")


@app.command()
def eval_stage2(
    data_json: Path,
    savedir: Path,
    artifacts: Path,
    res: int,
    checkpoint: Path,
    dset_min: float,
    dset_max: float,
    device: str = "cuda",
    iters: int = 8,
    search_range: int = 3,
    patch_factor: int = 4,
    split="val",
    diffeomorphic: bool = True,
):
    """
    Stage2 eval
    """

    with open(data_json, "r") as f:
        if split not in json.load(f):
            return

    dataset = PatchDatasetWithArtifacts(
        data_json,
        res,
        artifacts,
        patch_factor,
        split=split,
        dset_min=dset_min,
        dset_max=dset_max,
    )

    savedir.mkdir(exist_ok=True)

    if search_range == 0:
        model = SomeNetNoCorr(iters=iters, diffeomorphic=diffeomorphic).to(device)
    else:
        model = SomeNet(
            search_range=search_range, iters=iters, diffeomorphic=diffeomorphic
        ).to(device)

    model.load_state_dict(torch.load(checkpoint))

    for i in trange(0, len(dataset), dataset.n_patches * dataset.chan_split):
        flows, hiddens = defaultdict(list), defaultdict(list)
        for j in range(i, i + (dataset.n_patches * dataset.chan_split)):
            data = dataset[j]
            fixed, moving = data["fixed_image"], data["moving_image"]
            assert isinstance(fixed, torch.Tensor)
            assert isinstance(moving, torch.Tensor)
            fixed, moving = fixed.unsqueeze(0).to(device), moving.unsqueeze(0).to(
                device
            )
            with torch.no_grad(), evaluating(model):
                flow, hidden = model(fixed, moving)

            savename = f'{data["moving_image_name"]}2{data["fixed_image_name"]}.pt'
            flowin = data["flowin"]
            assert isinstance(flowin, torch.Tensor)
            flowin = flowin.squeeze().unsqueeze(0)
            flow = concat_flow(flow, flowin.to(device))

            flows[savename].append(
                (data["chan_index"], data["patch_index"], flow.detach().cpu())
            )
            hiddens[savename].append(
                (data["chan_index"], data["patch_index"], hidden.detach().cpu())
            )

        assert len(list(flows.keys())) == 1
        for k, v in flows.items():
            fchannels = defaultdict(list)
            hchannels = defaultdict(list)

            for i in v:
                fchannels[i[0]].append((i[1], i[2]))

            ffchannels = [
                torch.stack([i[1] for i in sorted(fchan, key=lambda x: x[0])], dim=-1)
                for fchan in fchannels.values()
            ]
            fk = torch.cat(ffchannels, dim=2)

            hv = hiddens[k]
            for i in hv:
                hchannels[i[0]].append((i[1], i[2]))

            fhchannels = [
                torch.stack([i[1] for i in sorted(hchan, key=lambda x: x[0])], dim=-1)
                for hchan in hchannels.values()
            ]

            hk = torch.cat(fhchannels, dim=2)

            pshape = fk.shape[-3:-1]
            res_shape = (fk.shape[2], *[i * 2 for i in pshape])

            fk = fk.squeeze(0)
            fk = einops.rearrange(fk, "c h d w p -> c (h d w) p")
            folded_flow = F.fold(fk, res_shape[-2:], pshape, stride=pshape)

            hk = hk.squeeze(0)
            hk = einops.rearrange(hk, "c h d w p -> c (h d w) p")

            moving, fixed = k.split("z2")
            moving = moving + "z"
            fixed = ".".join(fixed.split(".")[:-1])  # remove .pt

            folded_hidden = F.fold(hk, res_shape[-2:], pshape, stride=pshape)

            # torch.save(fk, savedir / (disp_name))
            torch.save(folded_flow, savedir / ("flow-" + k))
            torch.save(folded_hidden, savedir / ("hidden-" + k))


@app.command()
def eval_stage1(
    data_json: Path,
    savedir: Path,
    res: int,
    dset_min: float,
    dset_max: float,
    checkpoint: Path,
    device: str = "cuda",
    iters: int = 12,
    search_range: int = 3,
    split="val",
    patch_factor: int = 4,
    diffeomorphic: bool = True,
):
    """
    Stage1 eval
    """

    with open(data_json, "r") as f:
        if split not in json.load(f):
            return

    dataset = PatchDataset(
        data_json, res, patch_factor, split=split, dset_min=dset_min, dset_max=dset_max
    )

    savedir.mkdir(exist_ok=True)

    if search_range == 0:
        model = SomeNetNoCorr(iters=iters, diffeomorphic=diffeomorphic).to(device)
    else:
        model = SomeNet(
            search_range=search_range, iters=iters, diffeomorphic=diffeomorphic
        ).to(device)

    model.load_state_dict(torch.load(checkpoint))

    with evaluating(model):
        for i in trange(0, len(dataset), dataset.n_patches):
            flows, hiddens = defaultdict(list), defaultdict(list)
            for j in range(i, i + dataset.n_patches):
                data = dataset[j]
                fixed, moving = data["fixed_image"], data["moving_image"]
                assert isinstance(fixed, torch.Tensor)
                assert isinstance(moving, torch.Tensor)
                fixed, moving = fixed.unsqueeze(0).to(device), moving.unsqueeze(0).to(
                    device
                )
                with torch.no_grad():
                    flow, hidden = model(fixed, moving)
                savename = f'{data["moving_image_name"]}2{data["fixed_image_name"]}.pt'

                flows[savename].append((data["patch_index"], flow.detach().cpu()))
                hiddens[savename].append((data["patch_index"], hidden.detach().cpu()))

            assert len(flows.keys()) == 1, "Expected only one key"
            for k, v in flows.items():
                fk = torch.stack([i[1] for i in sorted(v, key=lambda x: x[0])], dim=-1)
                hk = torch.stack(
                    [i[1] for i in sorted(hiddens[k], key=lambda x: x[0])], dim=-1
                )

                fk = fk[..., 0]
                hk = hk[..., 0]
                torch.save(fk, savedir / ("flow-" + k))
                torch.save(hk, savedir / ("hidden-" + k))

            del flows, hiddens


def run_model_with_artifiacts_and_get_losses(
    model,
    data,
    *,
    res,
    device,
    image_loss,
    image_loss_weight,
    reg_loss_weight,
    seg_loss_weight,
    writer=None,
    step=None,
):
    fixed, moving, hidden = (
        data["fixed_image"],
        data["moving_image"],
        data["hidden"],
    )

    fixed, moving, hidden = (
        fixed.to(device),
        moving.to(device),
        hidden.to(device),
    )

    flow, hidden = model(fixed, moving, hidden)

    moved = warp_image(flow, moving)

    losses_dict: Dict[str, torch.Tensor] = {}
    losses_dict["grad"] = reg_loss_weight * Grad()(flow)

    losses_dict["image_loss"] = image_loss_weight * image_loss(
        moved.squeeze(), fixed.squeeze()
    )

    if "fixed_segmentation" in data:
        fixed_segmentation = data["fixed_segmentation"].to(device).float()
        moving_segmentation = data["moving_segmentation"].to(device).float()

        losses_dict["dice_loss"] = seg_loss_weight * DiceLoss()(
            fixed_segmentation, moving_segmentation, flow
        )

    if "fixed_keypoints" in data:
        flowin = data["flowin"].to(device)
        flow = concat_flow(flow, flowin)
        losses_dict["keypoints"] = res * TotalRegistrationLoss()(
            fixed_landmarks=data["fixed_keypoints"].squeeze(0),
            moving_landmarks=data["moving_keypoints"].squeeze(0),
            displacement_field=flow,
            fixed_spacing=data["fixed_spacing"].squeeze(0),
            moving_spacing=data["moving_spacing"].squeeze(0),
        )

    if model.training:
        for lk, lv in losses_dict.items():
            if not lv.requires_grad:
                raise Exception(
                    f"{lk} does not require gradient and will not affect learning"
                )

    return losses_dict


@app.command()
def train_with_artifacts(
    data_json: Path,
    checkpoint_dir: Path,
    artifacts: Path,
    dset_min: float,
    dset_max: float,
    res: int,
    patch_factor: int,
    steps: int,
    lr: float,
    image_loss_weight: float,
    reg_loss_weight: float,
    seg_loss_weight: float,
    log_freq: int,
    save_freq: int,
    val_freq: int,
    start: Optional[Path] = None,
    starting_step: Optional[int] = None,
    device: str = "cuda",
    image_loss_fn: str = "mse",
    iters: int = 2,
    search_range: int = 1,
    use_mask: bool = False,
    diffeomorphic: bool = False,
    num_workers: int = 4,
    noisy: bool = False,
):

    train_dataset = PatchDatasetWithArtifacts(
        data_json,
        res,
        artifacts,
        patch_factor,
        split="train",
        dset_min=dset_min,
        dset_max=dset_max,
    )
    val_dataset = PatchDatasetWithArtifacts(
        data_json,
        res,
        artifacts,
        patch_factor,
        split="val",
        dset_min=dset_min,
        dset_max=dset_max,
    )

    train_loader = InfiniteDataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )

    checkpoint_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=checkpoint_dir)

    if search_range == 0:
        model = SomeNetNoCorr(iters=iters, diffeomorphic=diffeomorphic).to(device)
    else:
        if noisy:
            modelclass = SomeNetNoisy
        else:
            modelclass = SomeNet

        model = modelclass(
            search_range=search_range, iters=iters, diffeomorphic=diffeomorphic
        ).to(device)

    step_count = 0
    if start is not None:
        model.load_state_dict(torch.load(start))

    if starting_step is not None:
        step_count = starting_step

    image_loss = get_loss_fn(image_loss_fn)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    run = partial(
        run_model_with_artifiacts_and_get_losses,
        res=res,
        device=device,
        image_loss=image_loss,
        image_loss_weight=image_loss_weight,
        reg_loss_weight=reg_loss_weight,
        seg_loss_weight=seg_loss_weight,
    )

    print(f"Dataset size is {len(train_dataset)}")
    print(f"Starting training from step {step_count}")
    for step_count, data in zip(trange(step_count, steps), train_loader):
        if step_count % log_freq == 0:
            losses_dict = run(model, data, writer=writer, step=step_count)
        else:
            losses_dict = run(model, data)
        total_loss = sum(losses_dict.values())
        assert isinstance(total_loss, torch.Tensor)
        losses_dict_log = {k: v.item() for k, v in losses_dict.items()}

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if val_freq > 0 and step_count % val_freq == 0 and step_count > 0:
            losses_cum_dict = defaultdict(list)
            with torch.no_grad(), evaluating(model):
                for data in val_loader:
                    losses = run(model, data)
                    for k, v in losses.items():
                        losses_cum_dict[k].append(v.item())

            for k, v in losses_cum_dict.items():
                writer.add_scalar(f"val_{k}", np.mean(v).item(), global_step=step_count)

        if step_count % save_freq == 0:
            torch.save(
                model.state_dict(),
                checkpoint_dir / f"rnn{res}x_{step_count}.pth",
            )

    torch.save(model.state_dict(), checkpoint_dir / f"rnn{res}x_{steps}.pth")


def run_model_and_get_losses(
    model,
    data,
    *,
    res,
    device,
    image_loss,
    image_loss_weight,
    reg_loss_weight,
    seg_loss_weight,
    writer=None,
    step=None,
):
    losses = {}
    fixed, moving = data["fixed_image"], data["moving_image"]
    fixed, moving = fixed.to(device), moving.to(device)
    flow, _ = model(fixed, moving)
    moved = warp_image(flow, moving)

    losses["image_loss"] = (
        image_loss_weight * image_loss(moved.squeeze(), fixed.squeeze())
    )
    losses["grad"] = (reg_loss_weight * Grad()(flow))

    if "fixed_keypoints" in data:
        losses["keypoints"] = (
            res
            * TotalRegistrationLoss()(
                fixed_landmarks=data["fixed_keypoints"].squeeze(0),
                moving_landmarks=data["moving_keypoints"].squeeze(0),
                displacement_field=flow,
                fixed_spacing=data["fixed_spacing"].squeeze(0),
                moving_spacing=data["moving_spacing"].squeeze(0),
            )
        )

    if "fixed_segmentation" in data:
        fixed_segmentation = data["fixed_segmentation"].to(device).float()
        moving_segmentation = data["moving_segmentation"].to(device).float()

        losses["dice_loss"] = (
            seg_loss_weight * DiceLoss()(fixed_segmentation, moving_segmentation, flow)
        )


    if model.training:
        for lk, lv in losses.items():
            if not lv.requires_grad:
                raise Exception(
                    f"{lk} does not require gradient and will not affect learning"
                )

    if writer != None and step != None:
        tb_log(
            writer,
            losses,
            step=step,
            moving_fixed_moved=(moving, fixed, moved),
        )

    return losses


@app.command()
def train(
    data_json: Path,
    checkpoint_dir: Path,
    res: int,
    patch_factor: int,
    iters: int,
    steps: int,
    dset_min: float,
    dset_max: float,
    lr: float,
    image_loss_fn: str,
    image_loss_weight: float,
    reg_loss_weight: float,
    seg_loss_weight: float,
    log_freq: int,
    save_freq: int,
    val_freq: int,
    start: Optional[Path] = None,
    use_mask: bool = False,
    diffeomorphic: bool = True,
    search_range: int = 3,
    num_workers: int = 4,
    starting_step: Optional[int] = None,
    noisy: bool = False,
    device: str = "cuda",
):
    """
    Trains the model at a specific resolution without any artifacts
    """

    train_dataset = PatchDataset(
        data_json,
        res,
        patch_factor,
        split="train",
        switch=True,
        dset_min=dset_min,
        dset_max=dset_max,
    )
    val_dataset = PatchDataset(
        data_json, res, patch_factor, split="val", dset_min=dset_min, dset_max=dset_max
    )

    train_loader = InfiniteDataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=True, num_workers=num_workers
    )

    checkpoint_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=checkpoint_dir)

    image_loss = get_loss_fn(image_loss_fn)

    if search_range == 0:
        model = SomeNetNoCorr(iters=iters, diffeomorphic=diffeomorphic).to(device)
    else:
        if noisy:
            modelclass = SomeNetNoisy
        else:
            modelclass = SomeNet

        model = modelclass(
            search_range=search_range, iters=iters, diffeomorphic=diffeomorphic
        ).to(device)
    step_count = 0
    if start is not None:
        model.load_state_dict(torch.load(start))

    if starting_step is not None:
        step_count = starting_step

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    run = partial(
        run_model_and_get_losses,
        res=res,
        device=device,
        image_loss=image_loss,
        image_loss_weight=image_loss_weight,
        reg_loss_weight=reg_loss_weight,
        seg_loss_weight=seg_loss_weight,
    )

    print(f"Starting training from step {step_count}")
    for step_count, data in zip(trange(step_count, steps), train_loader):
        if step_count % log_freq == 0:
            losses_dict = run(model, data, writer=writer, step=step_count)
        else:
            losses_dict = run(model, data)

        total_loss = sum(losses_dict.values())
        assert isinstance(total_loss, torch.Tensor)
        losses_dict_log = {k: v.item() for k, v in losses_dict.items()}

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        if val_freq > 0 and step_count % val_freq == 0:
            losses_cum_dict = defaultdict(list)
            with torch.no_grad(), evaluating(model):
                for data in val_loader:
                    losses = run(model, data)
                    for k, v in losses.items():
                        losses_cum_dict[k].append(v.item())

            for k, v in losses_cum_dict.items():
                writer.add_scalar(f"val_{k}", np.mean(v).item(), global_step=step_count)

        if step_count % save_freq == 0:
            torch.save(
                model.state_dict(),
                checkpoint_dir / f"rnn{res}x_{step_count}.pth",
            )

    torch.save(model.state_dict(), checkpoint_dir / f"rnn{res}x_{steps}.pth")


if __name__ == "__main__":
    app()
