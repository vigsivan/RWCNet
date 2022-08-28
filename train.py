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
from networks import SomeNet

app = Typer()


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
        cache_file: Path = Path("./stage2.pkl"),
        cache_patches_dir: Optional[Path] = None,
        precache: bool = False,
        diffeomorphic: bool=False
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
        self.diffeomorphic = diffeomorphic
        self.split = split

        if not cache_file.exists():
            cache = self.get_dataset_minmax_(data_json.name)
            with open(cache_file, "wb") as f:
                pickle.dump(cache, f)
        else:
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            if data_json.name in cache:
                self.min_int = cache[data_json.name]["min_int"]
                self.max_int = cache[data_json.name]["max_int"]
            else:
                cache_ = self.get_dataset_minmax_(data_json.name)
                cache.update(cache_)
                with open(cache_file, "wb") as f:
                    pickle.dump(cache, f)

        self.cache = cache[data_json.name]

        self.cache_patches_dir = cache_patches_dir
        self.cached = []
        if cache_patches_dir is not None:
            cache_patches_dir.mkdir(exist_ok=True)
            self.cached = [False] * len(self.indexes)
            if precache:
                self.precache()

    def get_dataset_minmax_(self, json_name):
        cache = defaultdict(dict)
        min_int, max_int = np.inf, -1 * np.inf

        f, m = "fixed", "moving"

        min_int, max_int = np.inf, -1 * np.inf
        for dat in self.data:
            fixed_image = Path(dat[f"{f}_image"])
            moving_image = Path(dat[f"{m}_image"])

            fixed_nib = nib.load(fixed_image)
            moving_nib = nib.load(moving_image)

            mi = min(fixed_nib.get_fdata().min(), moving_nib.get_fdata().min())
            ma = max(fixed_nib.get_fdata().max(), moving_nib.get_fdata().max())

            if mi < min_int:
                min_int = mi
            if ma > max_int:
                max_int = ma

        cache[json_name]["min_int"] = min_int
        cache[json_name]["max_int"] = max_int
        return cache

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
        r = self.res_factor
        p = self.patch_factor

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

        fixed = (fixed - self.cache["min_int"]) / (
            self.cache["max_int"] - self.cache["min_int"]
        )
        moving = (moving - self.cache["min_int"]) / (
            self.cache["max_int"] - self.cache["min_int"]
        )

        fixed = F.interpolate(fixed.unsqueeze(0), rshape).squeeze(0).float()
        moving = F.interpolate(moving.unsqueeze(0), rshape).squeeze(0).float()

        factor = rshape[-1] // flow.shape[-1]

        flow = F.interpolate(flow, rshape) * factor
        hidden = F.interpolate(hidden, rshape) * factor

        if self.diffeomorphic:
            scale = 1 / (2**7)
            flow = scale * flow
            for _ in range(7):
                flow = concat_flow(flow, flow)

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

            fixed_seg = F.interpolate(fixed_seg.unsqueeze(0), rshape, mode='nearest').squeeze(0).float()
            moving_seg = (
                F.interpolate(moving_seg.unsqueeze(0), rshape, mode='nearest').squeeze(0).float()
            )

            moving_seg = warp_image(flow.unsqueeze(0), moving_seg.unsqueeze(0)).squeeze(0)

            ret["fixed_segmentation"] = fixed_seg
            ret["moving_segmentation"] = moving_seg

        if "fixed_mask" in data:

            fixed_mask_nib = nib.load(data["fixed_mask"])
            moving_mask_nib = nib.load(data["moving_mask"])

            fixed_mask = torch.from_numpy(fixed_mask_nib.get_fdata()).unsqueeze(0)
            moving_mask = torch.from_numpy(moving_mask_nib.get_fdata()).unsqueeze(0)


            fixed_mask = F.interpolate(fixed_mask.unsqueeze(0), rshape, mode='nearest').squeeze(0).float()
            moving_mask = (
                F.interpolate(moving_mask.unsqueeze(0), rshape, mode='nearest').squeeze(0).float()
            )

            moving_mask = warp_image(flow.unsqueeze(0), moving_mask.unsqueeze(0)).squeeze(0)

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

        fixed_p = self.get_patch(
            fixed, 1, rshape, pshape, patch_index, chan_index
        )
        moving_p = self.get_patch(
            moving, 1, rshape, pshape, patch_index, chan_index
        )
        hidden_p = self.get_patch(
            hidden, hidden.shape[0], rshape, pshape, patch_index, chan_index
        )
        flow_p = self.get_patch(
            flow, 3, rshape, pshape, patch_index, chan_index
        )

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

            fixed_seg, moving_seg = data["fixed_segmentation"], data["moving_segmentation"]

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

            fixed_mask_p = self.get_patch(
                fixed_mask, 1, rshape, pshape, patch_index, chan_index
            )
            moving_mask_p = self.get_patch(
                moving_mask, 1, rshape, pshape, patch_index, chan_index
            )

            ret["fixed_mask"] = fixed_mask_p.long()
            ret["moving_mask"] = moving_mask_p.long()

        return ret

    def precache(self):
        print("Precaching...")
        index = 0
        for data_index, data in tqdm(enumerate(self.data)):
            loaded_data = None
            for chan_index in range(self.chan_split):
                for patch_index in range(self.n_patches):
                    assert self.cache_patches_dir is not None
                    cachefile = self.cache_patches_dir / f"{self.split}_{index}.pkl"
                    if cachefile.exists():
                        self.cached[index] = True
                        index += 1
                        continue
                    if loaded_data is None:
                        loaded_data = self.load_data_(data_index)
                    patch_data = self.get_patch_data(loaded_data, chan_index, patch_index)
                    with open(cachefile, "wb") as f:
                        pickle.dump(patch_data, f)
                    self.cached[index] = True
                    index += 1


    def __getitem__(self, index: int):
        if self.cache_patches_dir is not None and self.cached[index]:
            with open(self.cache_patches_dir / f"{self.split}_{index}.pkl", "rb") as cached_file:
                ret = pickle.load(cached_file)
            return ret

        data_index, chan_index, patch_index = self.indexes[index]
        data = self.load_data_(data_index)

        patch_data = self.get_patch_data(data, chan_index, patch_index)

        if self.cache_patches_dir is not None:
            with open(self.cache_patches_dir / f"{self.split}_{index}.pkl", "wb") as cached_file:
                pickle.dump(patch_data, cached_file)
            self.cached[index] = True

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
        cache_file: Path = Path("./stage2.pkl"),
    ):
        super().__init__()

        assert patch_factor == res_factor, "Stage 1 training assumes patch factor equals res factor"
        if res_factor not in [1, 2, 4]:
            raise ValueError(f"Unacceptable resolution factor {res_factor}")

        with open(data_json, "r") as f:
            data = json.load(f)[split]

        self.data = data
        self.indexes = [(i, 0) for i, _ in enumerate(data)]
        self.res_factor = res_factor
        self.patch_factor = patch_factor
        self.n_patches = 1

        if not cache_file.exists():
            cache = self.get_dataset_minmax_(data_json.name)
            with open(cache_file, "wb") as f:
                pickle.dump(cache, f)
        else:
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            if data_json.name in cache:
                self.min_int = cache[data_json.name]["min_int"]
                self.max_int = cache[data_json.name]["max_int"]
            else:
                cache_ = self.get_dataset_minmax_(data_json.name)
                cache.update(cache_)
                with open(cache_file, "wb") as f:
                    pickle.dump(cache, f)

        self.cache = cache[data_json.name]

    def get_dataset_minmax_(self, json_name):
        cache = defaultdict(dict)
        min_int, max_int = np.inf, -1 * np.inf

        f, m = "fixed", "moving"

        min_int, max_int = np.inf, -1 * np.inf
        for dat in self.data:
            fixed_image = Path(dat[f"{f}_image"])
            moving_image = Path(dat[f"{m}_image"])

            fixed_nib = nib.load(fixed_image)
            moving_nib = nib.load(moving_image)

            mi = min(fixed_nib.get_fdata().min(), moving_nib.get_fdata().min())
            ma = max(fixed_nib.get_fdata().max(), moving_nib.get_fdata().max())

            if mi < min_int:
                min_int = mi
            if ma > max_int:
                max_int = ma

        cache[json_name]["min_int"] = min_int
        cache[json_name]["max_int"] = max_int
        return cache

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

        fixed = (fixed - self.cache["min_int"]) / (
            self.cache["max_int"] - self.cache["min_int"]
        )
        moving = (moving - self.cache["min_int"]) / (
            self.cache["max_int"] - self.cache["min_int"]
        )

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
            fname, mname = Path(data[f"{f}_segmentation"]), Path(data[f"{m}_segmentation"])

            fixed_nib = nib.load(fname)
            moving_nib = nib.load(mname)

            fixed = torch.from_numpy(fixed_nib.get_fdata()).unsqueeze(0)
            moving = torch.from_numpy(moving_nib.get_fdata()).unsqueeze(0)

            fixed = (fixed - self.cache["min_int"]) / (
                self.cache["max_int"] - self.cache["min_int"]
            )
            moving = (moving - self.cache["min_int"]) / (
                self.cache["max_int"] - self.cache["min_int"]
            )

            fixed = F.interpolate(fixed.unsqueeze(0), rshape).squeeze(0).float()
            moving = F.interpolate(moving.unsqueeze(0), rshape).squeeze(0).float()

            ret["fixed_segmentation"] = fixed
            ret["moving_segmentation"] = moving

        if "fixed_mask" in data:
            fname, mname = Path(data[f"{f}_mask"]), Path(data[f"{m}_mask"])

            fixed_nib = nib.load(fname)
            moving_nib = nib.load(mname)

            fixed = torch.from_numpy(fixed_nib.get_fdata()).unsqueeze(0)
            moving = torch.from_numpy(moving_nib.get_fdata()).unsqueeze(0)

            fixed = F.interpolate(fixed.unsqueeze(0), rshape).squeeze(0).float()
            moving = F.interpolate(moving.unsqueeze(0), rshape).squeeze(0).float()

            ret["fixed_mask"] = fixed
            ret["moving_mask"] = moving

        return ret


@app.command()
def eval_stage3(
    data_json: Path,
    savedir: Path,
    artifacts: Path,
    res: int,
    checkpoint: Path,
    device: str = "cuda",
    iters: int = 4,
    search_range: int = 2,
    split: str = "val",
):
    """
    Stage3 (Final) eval
    """

    dataset = PatchDatasetStage2(data_json, res, artifacts, 2, split="train")
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
            savename = (
                f'{data["moving_image_name"][0]}2{data["fixed_image_name"][0]}.pt'
            )
            flows[savename].append((data["patch_index"], flow.detach().cpu()))

            flow, _ = model(moving, fixed)
            flowin = data["flowin"].to(device)
            flow = concat_flow(flow, flowin)

            savename = (
                f'{data["fixed_image_name"][0]}2{data["moving_image_name"][0]}.pt'
            )
            flows[savename].append((data["patch_index"], flow.detach().cpu()))

        for k, v in flows.items():
            try:
                fk = torch.stack([i[1] for i in sorted(v)], dim=-1)
                pshape = fk.shape[-3:-1]
                res_shape = (fk.shape[2], *[i * 4 for i in pshape])
                inp = fk.squeeze(0)
                inp = einops.rearrange(inp, "c h d w p -> c (h d w) p")
                folded = F.fold(inp, res_shape[-2:], pshape, stride=pshape)
                folded = folded.unsqueeze(0)
                moving, fixed = k.split("z2")
                moving = moving + "z"
                fixed = ".".join(fixed.split(".")[:-1])  # remove .pt
                disp_name = f"disp_{fixed[-16:-12]}_{moving[-16:-12]}"
                disp_np = torch2skimage_disp(folded)
                disp_nib = nib.Nifti1Image(disp_np, affine=np.eye(4))
                nib.save(disp_nib, savedir / f"{disp_name}.nii.gz")
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
    iters: int = 8,
    search_range: int = 3,
    split="val",
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
                flow, hidden = model(fixed, moving)
                savename = f'{data["moving_image_name"]}2{data["fixed_image_name"]}.pt'
                flowin = data["flowin"]
                assert isinstance(flowin, torch.Tensor)
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
                    torch.stack(
                        [i[1] for i in sorted(fchan, key=lambda x: x[0])], dim=-1
                    )
                    for fchan in fchannels.values()
                ]
                fk = torch.cat(ffchannels, dim=2)

                hv = hiddens[k]
                for i in hv:
                    hchannels[i[0]].append((i[1], i[2]))

                fhchannels = [
                    torch.stack(
                        [i[1] for i in sorted(hchan, key=lambda x: x[0])], dim=-1
                    )
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
                folded_hidden = F.fold(hk, res_shape[-2:], pshape, stride=pshape)

                torch.save(folded_flow, savedir / ("flow-" + k))
                torch.save(folded_hidden, savedir / ("hidden-" + k))


@app.command()
def eval_stage1(
    data_json: Path,
    savedir: Path,
    res: int,
    checkpoint: Path,
    device: str = "cuda",
    iters: int = 12,
    search_range: int = 3,
    split="val",
):
    """
    Stage1 eval
    """

    dataset = PatchDataset(data_json, res, 4, split=split)

    savedir.mkdir(exist_ok=True)

    model = SomeNet(iters=iters, search_range=search_range).to(device)
    model.load_state_dict(torch.load(checkpoint))

    with torch.no_grad(), evaluating(model):
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


@app.command()
def train_stage2(
    data_json: Path,
    checkpoint_dir: Path,
    artifacts: Path,
    res: int,
    patch_factor: int=4,
    start: Optional[Path] = None,
    steps: int = 10000,
    lr: float = 3e-4,
    device: str = "cuda",
    image_loss_weight: float = 1,
    reg_loss_weight: float = 0.05,
    seg_loss_weight: float = 0.1,
    kp_loss_weight: float = 1,
    log_freq: int = 100,
    save_freq: int = 100,
    val_freq: int = 1000,
    iters: int = 2,
    search_range: int = 1,
    cache_dir: Optional[Path] = None,
    use_mask: bool = False,
    diffeomorphic: bool=False,
):
    """
    Stage2 training
    """

    if cache_dir is not None:
        precache = True
    else:
        precache=False

    train_dataset = PatchDatasetStage2(
            data_json, res, artifacts, patch_factor, split="train", cache_patches_dir=cache_dir, precache=precache, diffeomorphic=diffeomorphic
    )
    val_dataset = PatchDatasetStage2(
        data_json, res, artifacts, patch_factor, split="val", cache_patches_dir=cache_dir, precache=precache, diffeomorphic=diffeomorphic
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, #num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, #num_workers=4
    )

    checkpoint_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=checkpoint_dir)

    model = SomeNet(search_range=search_range, iters=iters, diffeomorphic=diffeomorphic).to(device)
    step_count = 0
    if start is not None:
        model.load_state_dict(torch.load(start))
        step_count = int(start.name.split("_")[-1].split(".")[0])

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Dataset size is {len(train_dataset)}")
    print(f"Starting training from step {step_count}")
    while step_count + 1 < steps:
        for step_count, data in zip(trange(step_count, steps), train_loader):
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

            if "fixed_mask" in data and use_mask:
                fixed_mask = data["fixed_mask"].to(device).float()
                moving_mask = data["moving_mask"].to(device).float()

                moved_mask = warp_image(flow, moving_mask)

                fixed = fixed_mask * fixed
                moved = moved_mask * moved
                moving = moving_mask * moving

            losses_dict["image_loss"] = image_loss_weight * MutualInformationLoss()(
                moved.squeeze(), fixed.squeeze()
            )

            if "fixed_segmentation" in data:
                fixed_segmentation = data["fixed_segmentation"].to(device).float()
                moving_segmentation = data["moving_segmentation"].to(device).float()

                moved_segmentation = warp_image(flow, moving_segmentation)

                fixed_segmentation = torch.round(fixed_segmentation)
                moved_segmentation = torch.round(moved_segmentation)

                losses_dict["dice_loss"] = seg_loss_weight * DiceLoss()(fixed_segmentation, moved_segmentation)

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

                        if "fixed_mask" in data and use_mask:
                            fixed_mask = data["fixed_mask"].to(device).float()
                            moving_mask = data["moving_mask"].to(device).float()

                            moved_mask = warp_image(flow, moving_mask)

                            fixed = fixed_mask * fixed
                            moved = moved_mask * moved

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

                        if "fixed_keypoints" in data:
                            flowin = data["flowin"].to(device)
                            flow = concat_flow(flow, flowin)
                            losses_cum_dict["keypoints"].append(
                                res
                                * TotalRegistrationLoss()(
                                    fixed_landmarks=data["fixed_keypoints"].squeeze(0),
                                    moving_landmarks=data["moving_keypoints"].squeeze(
                                        0
                                    ),
                                    displacement_field=flow,
                                    fixed_spacing=data["fixed_spacing"].squeeze(0),
                                    moving_spacing=data["moving_spacing"].squeeze(0),
                                ).item()
                            )

                        if "fixed_segmentation" in data:
                            fixed_segmentation = data["fixed_segmentation"].to(device).float()
                            moving_segmentation = data["moving_segmentation"].to(device).float()

                            moved_segmentation = warp_image(flow, moving_segmentation)

                            fixed_segmentation = torch.round(fixed_segmentation)
                            moved_segmentation = torch.round(moved_segmentation)

                            losses_cum_dict["dice_loss"].append(
                                    seg_loss_weight
                                    * DiceLoss()(fixed_segmentation, moved_segmentation)
                            )


                for k, v in losses_cum_dict.items():
                    writer.add_scalar(
                        f"val_{k}", np.mean(v).item(), global_step=step_count
                    )

            if step_count % save_freq == 0:
                torch.save(
                    model.state_dict(),
                    checkpoint_dir / f"rnn{res}x_{step_count}.pth",
                )

    torch.save(model.state_dict(), checkpoint_dir / f"rnn{res}x_{step_count+1}.pth")


@app.command()
def train_stage1(
    data_json: Path,
    checkpoint_dir: Path,
    res: int,
    patch_factor: int=4,
    start: Optional[Path] = None,
    steps: int = 10000,
    lr: float = 3e-4,
    device: str = "cuda",
    image_loss_weight: float = 1,
    reg_loss_weight: float = 0.1,
    seg_loss_weight: float=.1,
    kp_loss_weight: float = 1,
    log_freq: int = 5,
    save_freq: int = 50,
    val_freq: int = 50,
    use_mask: bool=False,
    diffeomorphic: bool=False,
    search_range: int=3,
    iters: int=12,
):
    """
    Stage1 training
    """

    train_dataset = PatchDataset(data_json, res, patch_factor, split="train")
    val_dataset = PatchDataset(data_json, res, patch_factor, split="val")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)

    checkpoint_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=checkpoint_dir)

    model = SomeNet(search_range=search_range, iters=iters, diffeomorphic=diffeomorphic).to(device)
    step_count = 0
    if start is not None:
        model.load_state_dict(torch.load(start))
        step_count = int(start.name.split("_")[-1].split(".")[0])

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training from step {step_count}")
    while step_count + 1 < steps:
        for step_count, data in zip(trange(step_count, steps), train_loader):

            fixed, moving = data["fixed_image"], data["moving_image"]
            fixed, moving = fixed.to(device), moving.to(device)
            flow, hidden = model(fixed, moving)

            moved = warp_image(flow, moving)

            losses_dict: Dict[str, torch.Tensor] = {}

            if "fixed_mask" in data and use_mask:
                fixed_mask = data["fixed_mask"].to(device).float()
                moving_mask = data["moving_mask"].to(device).float()

                moved_mask = warp_image(flow, moving_mask)

                fixed = fixed_mask * fixed
                moved = moved_mask * moved
                moving = moving_mask * moving

            losses_dict["image_loss"] = image_loss_weight * MutualInformationLoss()(
                moved.squeeze(), fixed.squeeze()
            )
            losses_dict["grad"] = reg_loss_weight * Grad()(flow)

            if "fixed_keypoints" in data:
                losses_dict["keypoints"] = res * TotalRegistrationLoss()(
                    fixed_landmarks=data["fixed_keypoints"].squeeze(0),
                    moving_landmarks=data["moving_keypoints"].squeeze(0),
                    displacement_field=flow,
                    fixed_spacing=data["fixed_spacing"].squeeze(0),
                    moving_spacing=data["moving_spacing"].squeeze(0),
                )

            if "fixed_segmentation" in data:
                fixed_segmentation = data["fixed_segmentation"].to(device).float()
                moving_segmentation = data["moving_segmentation"].to(device).float()

                moved_segmentation = warp_image(flow, moving_segmentation)

                fixed_segmentation = torch.round(fixed_segmentation)
                moved_segmentation = torch.round(moved_segmentation)

                losses_dict["dice_loss"] = seg_loss_weight * DiceLoss()(
                    fixed_segmentation, moved_segmentation)

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
                                * MutualInformationLoss()(
                                    moved.squeeze(), fixed.squeeze()
                                )
                            ).item()
                        )
                        losses_cum_dict["grad"].append(
                            (reg_loss_weight * Grad()(flow)).item()
                        )

                        if "fixed_keypoints" in data:
                            losses_cum_dict["keypoints"].append(
                                res
                                * TotalRegistrationLoss()(
                                    fixed_landmarks=data["fixed_keypoints"].squeeze(0),
                                    moving_landmarks=data["moving_keypoints"].squeeze(
                                        0
                                    ),
                                    displacement_field=flow,
                                    fixed_spacing=data["fixed_spacing"].squeeze(0),
                                    moving_spacing=data["moving_spacing"].squeeze(0),
                                ).item()
                            )

                        if "fixed_segmentation" in data:
                            fixed_segmentation = data["fixed_segmentation"].to(device).float()
                            moving_segmentation = data["moving_segmentation"].to(device).float()

                            moved_segmentation = warp_image(flow, moving_segmentation)

                            fixed_segmentation = torch.round(fixed_segmentation)
                            moved_segmentation = torch.round(moved_segmentation)

                            losses_cum_dict["dice_loss"].append(
                                    seg_loss_weight
                                    * DiceLoss()(fixed_segmentation, moved_segmentation).item()
                            )


                for k, v in losses_cum_dict.items():
                    writer.add_scalar(
                        f"val_{k}", np.mean(v).item(), global_step=step_count
                    )

            if step_count % save_freq == 0:
                torch.save(
                    model.state_dict(),
                    checkpoint_dir / f"rnn{res}x_{step_count}.pth",
                )

    torch.save(model.state_dict(), checkpoint_dir / f"rnn{res}x_{step_count}.pth")


if __name__ == "__main__":
    app()
