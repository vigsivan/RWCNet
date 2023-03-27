import json
from pathlib import Path

import einops
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import trange

from common import (
    MINDSEG,
    MINDSSC,
    adam_optimization,
    concat_flow,
    load_keypoints,
    warp_image,
)
from config import EvalConfig
from differentiable_metrics import (
    MSE,
    DiceLoss,
    MutualInformationLoss,
    TotalRegistrationLoss,
)
from networks import SomeNet, SomeNetNoCorr

get_spacing = lambda x: np.sqrt(np.sum(x.affine[:3, :3] * x.affine[:3, :3], axis=0))


class EvalDataset(Dataset):
    """
    Normalizes inputs using cached intensities
    """

    def __init__(
        self,
        data_json: Path,
        split: str,
        min_int,
        max_int,
    ):
        super().__init__()

        with open(data_json, "r") as f:
            data = json.load(f)[split]

        self.data = data
        self.indexes = [(i, 0) for i, _ in enumerate(data)]
        self.n_patches = 1
        self.min_int = min_int
        self.max_int = max_int

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index: int):
        data_index, _ = self.indexes[index]
        data = self.data[data_index]

        f, m = "fixed", "moving"

        fname, mname = Path(data[f"{f}_image"]), Path(data[f"{m}_image"])

        fixed_nib = nib.load(fname)
        moving_nib = nib.load(mname)

        fixed = torch.from_numpy(fixed_nib.get_fdata()).unsqueeze(0)
        moving = torch.from_numpy(moving_nib.get_fdata()).unsqueeze(0)

        fixed = (fixed - self.min_int) / (self.max_int - self.min_int)
        moving = (moving - self.min_int) / (self.max_int - self.min_int)

        ret = {
            "fixed_image": fixed,
            "moving_image": moving,
        }

        ret["fixed_name"] = fname.name
        ret["moving_name"] = mname.name

        if "fixed_mask" in data:
            fixed_mask_nib = nib.load(data["fixed_mask"])
            moving_mask_nib = nib.load(data["moving_mask"])

            fixed_mask = (
                torch.from_numpy(fixed_mask_nib.get_fdata()).unsqueeze(0).float()
            )
            moving_mask = (
                torch.from_numpy(moving_mask_nib.get_fdata()).unsqueeze(0).float()
            )

            ret["fixed_mask"] = fixed_mask
            ret["moving_mask"] = moving_mask

        if "fixed_keypoints" in data:
            fixed_kps = load_keypoints(data["fixed_keypoints"])
            moving_kps = load_keypoints(data["moving_keypoints"])

            ret["fixed_keypoints"] = fixed_kps
            ret["moving_keypoints"] = moving_kps
            ret["fixed_spacing"] = torch.Tensor(get_spacing(fixed_nib))
            ret["moving_spacing"] = torch.Tensor(get_spacing(moving_nib))

        if "fixed_segmentation" in data:
            fixed_seg_nib = nib.load(data["fixed_segmentation"])
            moving_seg_nib = nib.load(data["moving_segmentation"])

            fixed_segmentation = (
                torch.from_numpy(fixed_seg_nib.get_fdata()).unsqueeze(0).long()
            )
            moving_segmentation = (
                torch.from_numpy(moving_seg_nib.get_fdata()).unsqueeze(0).long()
            )

            ret["fixed_segmentation"] = fixed_segmentation
            ret["moving_segmentation"] = moving_segmentation

        return ret


def get_patches(tensor: torch.Tensor, res_factor: int, patch_factor: int):
    r = res_factor
    p = patch_factor
    n_channels = tensor.shape[0]

    pshape = tuple(s * r // p for s in tensor.shape[-3:])

    assert len(tensor.shape) == 4, "Expected tensor to have four dimensions"
    tensor_ps = F.unfold(tensor, pshape[-2:], stride=pshape[-2:])

    L = tensor_ps.shape[-1]

    tensor_ps = tensor_ps.reshape(n_channels, tensor.shape[-3], *pshape[-2:], L)
    tensor_ps = torch.split(tensor_ps.unsqueeze(0), [pshape[0]] * int(p / r), dim=2)
    return tensor_ps


def run_model_no_patch(model, fixed, moving, flow_agg, hidden):
    with torch.no_grad():
        flow, hidden, fixed_feats, moving_feats = model(
            fixed, moving, hidden, ret_fmap=True
        )
    if flow_agg is None:
        flow_agg = flow
    else:
        flow_agg = concat_flow(flow, flow_agg)
    return flow_agg, hidden, fixed_feats, moving_feats


def fold_(t, res_shape, pshape):
    t = einops.rearrange(t.squeeze(0), "c h d w p -> c (h d w) p")
    folded_flow = F.fold(t, res_shape[-2:], pshape, stride=pshape)
    folded_flow = folded_flow.unsqueeze(0)
    return folded_flow


def run_model_with_patches(
    res_factor, patch_factor, model, fixed, moving, flow_agg, hidden
):
    r = res_factor
    p = patch_factor

    res_shape = fixed.shape[-3:]
    pshape = tuple(int(r / p * s) for s in res_shape)[-2:]

    flows = []
    hiddens = []
    fixed_feats = []
    moving_feats = []

    fixed_patches = get_patches(fixed.squeeze(0), r, p)
    moving_patches = get_patches(moving.squeeze(0), r, p)
    hidden_patches = None

    if hidden is not None:
        hidden_patches = get_patches(hidden.squeeze(0), r, p)

    n_patches = fixed_patches[0].shape[-1]

    for cindex in range(len(fixed_patches)):
        flows_p, hiddens_p = [], []
        fixed_ps, moving_ps = [], []
        for pindex in range(n_patches):
            with torch.no_grad():
                flow, hidden_p, fixed_p, moving_p = model(
                    fixed_patches[cindex][..., pindex],
                    moving_patches[cindex][..., pindex],
                    (
                        hidden_patches[cindex][..., pindex]
                        if hidden_patches is not None
                        else None
                    ),
                    ret_fmap=True,
                )
                flows_p.append(flow.detach())
                hiddens_p.append(hidden_p.detach())
                fixed_ps.append(fixed_p)
                moving_ps.append(moving_p)

        flows.append(torch.stack(flows_p, dim=-1))
        hiddens.append(torch.stack(hiddens_p, dim=-1))
        fixed_feats.append(torch.stack(fixed_ps, dim=-1))
        moving_feats.append(torch.stack(moving_ps, dim=-1))

    fk = torch.cat(flows, dim=2)
    hk = torch.cat(hiddens, dim=2)
    Ffk = torch.cat(fixed_feats, dim=2)
    Fmk = torch.cat(moving_feats, dim=2)

    folded_flow = fold_(fk, res_shape, pshape)
    folded_hidden = fold_(hk, res_shape, pshape)
    folded_fixed_f = fold_(Ffk, res_shape, pshape)
    folded_moving_f = fold_(Fmk, res_shape, pshape)

    if flow_agg is not None:
        flow_agg = concat_flow(folded_flow, flow_agg)
    else:
        flow_agg = folded_flow

    return flow_agg, folded_hidden, folded_fixed_f, folded_moving_f


def evaluate(data, flow_agg, fixed_res, moving_res, res):
    # FIXME: make the image loss configurable
    mi_loss = MSE()(fixed_res, moving_res)
    print("Mutual Information Loss: ", mi_loss.item())

    if "fixed_keypoints" in data:
        fixed_kps = data["fixed_keypoints"] / res
        moving_kps = data["moving_keypoints"] / res
        tre_loss = res * TotalRegistrationLoss()(
            fixed_landmarks=fixed_kps.squeeze(0),
            moving_landmarks=moving_kps.squeeze(0),
            displacement_field=flow_agg,
            fixed_spacing=data["fixed_spacing"].squeeze(0),
            moving_spacing=data["moving_spacing"].squeeze(0),
        )

    if "fixed_segmentation" in data:
        fixed_seg = (
            data["fixed_segmentation"]
            .long()
            .squeeze()
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
        )
        moving_seg = (
            data["moving_segmentation"]
            .long()
            .squeeze()
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
        )
        fixed_seg = F.interpolate(
            fixed_seg, tuple(s // res for s in fixed_seg.shape[-3:]), mode="nearest"
        )
        moving_seg = F.interpolate(
            moving_seg, tuple(s // res for s in moving_seg.shape[-3:]), mode="nearest"
        )
        dice_loss = DiceLoss()(
            fixed_seg.to(flow_agg.device), moving_seg.to(flow_agg.device), flow_agg
        )

        print("Dice Loss", dice_loss.item())


def eval(data_json: Path, eval_config: Path):
    with open(eval_config, "r") as f:
        config_dict = json.load(f)
        config = EvalConfig(**config_dict)

    config.save_path.mkdir(exist_ok=True)
    eval_dataset = EvalDataset(
        data_json=data_json,
        split=config.split,
        max_int=config.max_int,
        min_int=config.min_int,
    )
    for i in trange(len(eval_dataset)):
        data = eval_dataset[i]
        hidden = None
        flow_agg = None
        fixed_features = []
        moving_features = []

        fixed, moving = data["fixed_image"], data["moving_image"]
        assert isinstance(fixed, torch.Tensor)
        assert isinstance(moving, torch.Tensor)
        fixed, moving = (
            fixed.unsqueeze(0).float(),
            moving.unsqueeze(0).float(),
        )

        for stage in config.stages:
            r = stage.res_factor
            p = stage.patch_factor
            res_shape = tuple(s // r for s in fixed.shape[-3:])
            fixed_res = F.interpolate(fixed, res_shape).to(config.device)
            moving_res = F.interpolate(moving, res_shape).to(config.device)

            if flow_agg is not None:
                up_factor = res_shape[-1] / flow_agg.shape[-1]
                flow_agg = F.interpolate(flow_agg, res_shape) * up_factor
                hidden = F.interpolate(hidden, res_shape)
                moving_res = warp_image(flow_agg, moving_res)

            if stage.search_range == 0:
                model = SomeNetNoCorr(
                    iters=stage.iters, diffeomorphic=stage.diffeomorphic
                )
            else:
                model = SomeNet(
                    iters=stage.iters,
                    search_range=stage.search_range,
                    diffeomorphic=stage.diffeomorphic,
                )
            model.load_state_dict(torch.load(stage.checkpoint))
            model = model.eval().to(config.device)
            if r == p:
                (
                    flow_agg,
                    hidden,
                    fixed_features_res,
                    moving_features_res,
                ) = run_model_no_patch(model, fixed_res, moving_res, flow_agg, hidden)
                fixed_features.append(fixed_features_res)
                moving_features.append(moving_features_res)
            else:
                (
                    flow_agg,
                    hidden,
                    fixed_features_res,
                    moving_features_res,
                ) = run_model_with_patches(
                    r, p, model, fixed_res, moving_res, flow_agg, hidden
                )
                fixed_features.append(fixed_features_res)
                moving_features.append(moving_features_res)

            if config.eval_at_each_stage:
                evaluate(data, flow_agg, fixed_res, moving_res, r)

            del fixed_res, moving_res, model

        assert flow_agg is not None
        evaluate(data, flow_agg, fixed, moving, 1)
        # FIXME: instance optimization
        # fixed = fixed.to(config.device)
        # moving = moving.to(config.device)
        #
        # shape = fixed_features[-1].shape[-3:]
        # fixed_features = torch.cat([F.interpolate(f, shape) for f in fixed_features], dim=1).half()
        # moving_features = torch.cat([F.interpolate(f,shape) for f in moving_features], dim=1).half()
        #
        # # fixed_features = fixed_features[-1].half()
        # # moving_features = moving_features[-1].half()
        #
        # fixed_features, moving_features = MINDSSC(fixed,1,2, config.device).half(), MINDSSC(moving,1,2, config.device).half()
        #
        # fixed_features, moving_features = fixed.half(), moving.half()
        # if "fixed_mask" in data:
        #     fixed_mask = data["fixed_mask"]
        #     moving_mask = data["moving_mask"]
        #
        #     fixed_mask = fixed_mask.to(config.device).unsqueeze(0).float()
        #     moving_mask = moving_mask.to(config.device).unsqueeze(0).float()
        #
        #     fixed_features = fixed_mask * fixed_features
        #     moving_features = moving_mask * moving_features
        #
        # # if "fixed_segmentation" in data:
        # #     fixed_seg = data["fixed_mask"]
        # #     moving_seg = data["moving_mask"]
        # #
        # #     fixed_seg = fixed_seg.to(config.device).unsqueeze(0).float()
        # #     moving_seg = moving_seg.to(config.device).unsqueeze(0).float()
        # #
        # #     maxlabels = max(torch.unique(fixed_seg.long()).shape[0], torch.unique(moving_seg.long()).shape[0])
        # #
        # #     weight = 1 / (
        # #         torch.bincount(fixed_seg.long().reshape(-1), minlength=maxlabels)
        # #         + torch.bincount(moving_seg.long().reshape(-1), minlength=maxlabels)
        # #     ).float().pow(0.3)
        # #     weight[torch.isinf(weight)]=0.
        # #
        # #     fixed_features = MINDSEG(fixed_seg, norm_weight=weight)
        # #     moving_features = MINDSEG(fixed_seg, norm_weight=weight)
        #
        # io_shape = tuple(s//2 for s in fixed.shape[-3:])
        #
        #
        # iores = config.instance_opt_res
        # net = adam_optimization(
        #     mind_fixed=F.avg_pool3d(fixed_features, iores, stride=iores),
        #     mind_moving=F.avg_pool3d(moving_features, iores, stride=iores),
        #     disp=F.interpolate(flow_agg, io_shape) / iores,
        #     lambda_weight=1.25,
        #     norm=iores,
        #     image_shape= io_shape,
        #     iterations=100)
        #
        # disp_sample = F.avg_pool3d(
        #     F.avg_pool3d(net[0].weight, 3, stride=1, padding=1), 3, stride=1, padding=1
        # ).permute(0, 2, 3, 4, 1)
        #
        # fitted_grid = disp_sample.permute(0, 4, 1, 2, 3).detach()
        # disp_hr = F.interpolate(
        #     fitted_grid * iores,
        #     size=fixed.shape[-3:],
        #     mode="trilinear",
        #     align_corners=False,
        # )

        # evaluate(data, flow_agg, fixed, moving, 1)

        disp_np = flow_agg.detach().cpu().numpy()
        l2r_disp = einops.rearrange(disp_np.squeeze(), "t h w d -> h w d t")
        affine = np.eye(4)
        displacement_nib = nib.Nifti1Image(l2r_disp, affine=affine)
        fname, mname = data["fixed_name"], data["moving_name"]
        disp_name = f"disp_{fname[-16:-12]}_{mname[-16:-12]}.nii.gz"

        nib.save(displacement_nib, config.save_path / disp_name)


if __name__ == "__main__":
    import sys

    data_json = Path(sys.argv[1])
    eval_json = Path(sys.argv[2])
    eval(data_json, eval_json)
