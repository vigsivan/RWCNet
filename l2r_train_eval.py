import torch
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import json
from typing import Dict, Optional, List
import sys
import random
from pathlib import Path

from config import TrainConfig

from train import train_stage1, train_stage2, eval_stage1, eval_stage2

L2R_TRAIN_TIME_DATA = ["image", "mask"]
L2R_TEST_TIME_DATA = ["label", "keypoints"]

@dataclass
class SomeNetCheckpoint:
    stage: int
    step: int
    checkpoint: Optional[Path]

def find_last_checkpoint(config: TrainConfig, checkpointroot: Path) -> SomeNetCheckpoint:
    """
    Find the last checkpoint

    NOTE: if the config changes, i.e. previous stage number of steps has changed
    this function won't spot it. Possible to implement this function and retain
    old behaviour with a flag, but its probably better to avoid the situation above
    entirely by specifying a different save path.
    """
    num_stages = len(config.stages)
    get_step = lambda x: int(x.split('_')[-1].split('.')[0])
    for stage in range(num_stages, 0, -1):
        stage_folder = checkpointroot / f"stage{stage}"
        if stage_folder.exists():
            checkpoints = [f.name for f in stage_folder.iterdir() if f.name.endswith('.pth')]
            if len(checkpoints) == 0:
                # NOTE: we assume that the checkpoints from the previous stage were generated
                return SomeNetCheckpoint(
                    stage=stage,
                    step=0,
                    checkpoint=None
                )

                # continue
            last_checkpoint = max(checkpoints, key=get_step)
            return SomeNetCheckpoint(
                stage=stage,
                step=get_step(last_checkpoint),
                checkpoint=stage_folder / last_checkpoint
            )
    raise ValueError("Could not find folders for any of the stages")

def get_split_pairs_from_paired_dataset(data: Dict, root: Path):
    train_data = {}
    test_data = {}
    split_data = defaultdict(list)
    for dat in data["training"]:
        train_data[dat["image"]] = {
            (k if k != "label" else "segmentation"): str(root / v)
            for k, v in dat.items()
        }

    for dat in data["test"]:
        test_data[dat["image"]] = {
            (k if k != "label" else "segmentation"): str(root / v)
            for k, v in dat.items()
        }

    val_fixeds = [dat["fixed"] for dat in data["registration_val"]]

    for split_p in ["training"]:
        for dat in data[f"{split_p}_paired_images"]:
            split = "val" if dat["fixed"] in val_fixeds else "train"
            pair_dat = {}
            for lab in ["fixed", "moving"]:
                image = dat[lab]
                if image in train_data:
                    all_dat_for_image = train_data[image]
                    pair_dat.update(
                        {f"{lab}_{k}": v for k, v in all_dat_for_image.items()}
                    )
                else:
                    raise ValueError(f"{image} not found")
            else:
                split_data[split].append(pair_dat)

    # FIXME: their test specification is a mess
    # for dat in data[f"test_paired_images"]:
    #     pair_dat = {}
    #     for lab in ("fixed", "moving"):
    #         image = dat[lab]
    #         if image in train_data:
    #             all_dat_for_image = train_data[image]
    #             pair_dat.update(
    #                 {f"{lab}_{k}": v for k, v in all_dat_for_image.items()}
    #             )
    #         else:
    #             raise ValueError(f"{image} not found")
    #     else:
    #         split_data["test"].append(pair_dat)

    return split_data


def get_split_pairs_from_unpaired_dataset(data: Dict, root: Path):
    train_data = {}
    split_data = defaultdict(list)
    for dat in data["training"]:
        train_data[dat["image"]] = {
            (k if k != "label" else "segmentation"): str(root / v)
            for k, v in dat.items()
        }

    val_images = []
    for dat in data[f"registration_val"]:
        pair_dat = {}
        for lab in ("fixed", "moving"):
            image = dat[lab]
            if image in train_data:
                all_dat_for_image = train_data[image]
                pair_dat.update({f"{lab}_{k}": v for k, v in all_dat_for_image.items()})
            else:
                print(f"{image} not found")
                break
        else:
            split_data["val"].append(pair_dat)

    train_images = [train_data[k] for k in train_data.keys() if k not in val_images]
    training_pairs = get_pairs_from_list(train_images)
    split_data["train"] = training_pairs

    for dat in data[f"test_paired_images"]:
        pair_dat = {}
        for lab in ("fixed", "moving"):
            image = dat[lab]

    return split_data

def get_pairs_from_list(
    data: List[Dict[str, Path]], pairs_per_image: int = 1, root: Optional[Path] = None
) -> List[Dict[str, Path]]:
    return get_random_pairs_from_list(data, pairs_per_image)


def get_random_pairs_from_list(
    data: List[Dict[str, Path]], pairs_per_image: int = 5, seed: int=42
) -> List[Dict[str, Path]]:
    random.seed(seed)
    paired_data = []
    for i, di in enumerate(data):
        others = [random.choice(data) for _ in range(pairs_per_image)]
        for j, other in enumerate(others):
            pair_item = {}
            l1, l2 = "fixed", "moving"
            if j % 2 == 0:
                l1, l2 = l2, l1
            for k, v in di.items():
                pair_item[f"{l1}_{k}"] = v

            for k, v in other.items():
                pair_item[f"{l2}_{k}"] = v

            paired_data.append(pair_item)
    return paired_data


def get_split_pairs_from_hybrid_dataset(data: Dict, root: Path):
    train_data = {}
    test_data = {}
    split_data = defaultdict(list)
    for dat in data["training"]:
        train_data[dat["image"]] = {
            (k if k != "label" else "segmentation"): str(root / v)
            for k, v in dat.items()
        }

    for dat in data["test"]:
        test_data[dat["image"]] = {
            (k if k != "label" else "segmentation"): str(root / v)
            for k, v in dat.items()
        }

    val_images = []
    for dat in data[f"registration_val"]:
        pair_dat = {}
        for lab in ("fixed", "moving"):
            image = dat[lab]
            val_images.append(image)
            if image in train_data:
                all_dat_for_image = train_data[image]
                pair_dat.update({f"{lab}_{k}": v for k, v in all_dat_for_image.items()})
            else:
                print(f"{image} not found")
                break
        else:
            split_data["val"].append(pair_dat)

    train_paired_images = []
    for dat in data[f"training_paired_images"]:
        pair_dat = {}
        for lab in ("fixed", "moving"):
            image = dat[lab]
            train_paired_images.append(image)
            if image in train_data:
                all_dat_for_image = train_data[image]
                pair_dat.update({f"{lab}_{k}": v for k, v in all_dat_for_image.items()})
            else:
                print(f"{image} not found")
                break
        else:
            split_data["train"].append(pair_dat)

    train_paired_images = train_paired_images.extend(val_images)
    train_images = [
        train_data[k] for k in train_data.keys() if k not in train_paired_images
    ]
    training_pairs = get_pairs_from_list(train_images)
    split_data["train"].extend(training_pairs)

    for dat in data[f"test_paired_images"]:
        pair_dat = {}
        for lab in ("fixed", "moving"):
            image = dat[lab]
            if image in train_data:
                all_dat_for_image = train_data[image]
                pair_dat.update({f"{lab}_{k}": v for k, v in all_dat_for_image.items()})
            else:
                print(f"{image} not found")
                break
        else:
            split_data["test"].append(pair_dat)

    return split_data


def main(dataset_json: Path, config_json: Path):
    with open(dataset_json, "r") as f:
        data = json.load(f)

    with open(config_json, "r") as f:
        config_dict = json.load(f)
        config = TrainConfig(**config_dict)

    if config.gpu_num is not None:
        torch.cuda.set_device(config.gpu_num)
    root = dataset_json.parent
    metadata = {}
    training_recipe = {}

    metadata["name"] = data["name"]

    if data["tensorImageSize"]["0"] != "3D":
        raise ValueError("Only 3D image registration supported")

    if len(set(data["modality"].values())) > 2:
        raise ValueError(
            "This parse script does not support L2R's multimodal JSON format(mostly due to different parse format)."
        )

    if data["pairings"] == "paired":
        split_pairs = get_split_pairs_from_paired_dataset(data, root)

    elif data["pairings"] == "unpaired":
        split_pairs = get_split_pairs_from_unpaired_dataset(data, root)

    elif data["pairings"] == "hybrid":
        split_pairs = get_split_pairs_from_hybrid_dataset(data, root)
    else:
        raise ValueError(
            f"Expected pairings to be one of paired, unpaired or hybrid but it is {data['pairings']}"
        )

    image_shape = data["tensorImageShape"]["0"]
    if not all([s % 2 == 0 for s in image_shape]):
        raise ValueError(
            "Expect image dimensions to be at least multiple of 2. Please pad the inputs"
        )

    checkpointroot = Path("checkpoints")
    if config.savedir is not None:
        config.savedir.mkdir(exist_ok=True)
        checkpointroot = config.savedir / Path(f"checkpoints")

    resumption_point: Optional[SomeNetCheckpoint] = None
    if checkpointroot.exists() and (checkpointroot/"stage1").exists() and not config.overwrite:
        try:
            resumption_point = find_last_checkpoint(config, checkpointroot)
            print(f"Resuming from stage {resumption_point.stage-1} step {resumption_point.step}")
        except ValueError:
            resumption_point = None

    checkpointroot.mkdir(exist_ok=True)

    data_json = checkpointroot / f"data.json"

    if not data_json.exists():
        with open(data_json, "w") as f:
            json.dump(split_pairs, f)

    start = 0 if resumption_point is None else resumption_point.stage-1
    for i, stage in enumerate(config.stages[start:], start=start):
        if i == 0:
            print(f"Training stage {i}")
            train_stage1(
                data_json=data_json,
                checkpoint_dir=checkpointroot / f"stage{i+1}",
                steps=stage.steps,
                res=stage.res_factor,
                patch_factor=stage.patch_factor,
                iters=stage.iters,
                search_range=stage.search_range,
                diffeomorphic=config.diffeomorphic,
                save_freq=stage.save_freq,
                log_freq=stage.log_freq,
                val_freq=stage.val_freq,
                start=None if resumption_point is None else resumption_point.checkpoint,
                starting_step=None if resumption_point is None else resumption_point.step,
                noisy=config.noisy,
                noisy_v2=config.noisy_v2,
                image_loss_fn=config.image_loss_fn,
                image_loss_weight=config.image_loss_weight
            )
            resumption_point = None

            last_checkpoint = (checkpointroot
                    / f"stage{i+1}"
                    / f"rnn{stage.res_factor}x_{stage.steps}.pth")

            print(f"Evaluating stage {i}")
            for split in ("train", "val"):
                eval_stage1(
                    data_json=data_json,
                    savedir=checkpointroot / f"stage{i+1}_artifacts",
                    res=stage.res_factor,
                    iters=stage.iters,
                    search_range=stage.search_range,
                    patch_factor=stage.patch_factor,
                    checkpoint= last_checkpoint,
                    diffeomorphic=config.diffeomorphic,
                    split=split,
                )
        else:
            print(f"Training stage {i}")
            train_stage2(
                data_json=data_json,
                artifacts=checkpointroot / f"stage{i}_artifacts",
                checkpoint_dir=checkpointroot / f"stage{i+1}",
                steps=stage.steps,
                res=stage.res_factor,
                patch_factor=stage.patch_factor,
                iters=stage.iters,
                search_range=stage.search_range,
                diffeomorphic=config.diffeomorphic,
                start=None if resumption_point is None else resumption_point.checkpoint,
                save_freq=stage.save_freq,
                log_freq=stage.log_freq,
                val_freq=stage.val_freq,
                starting_step=None if resumption_point is None else resumption_point.step,
                noisy=config.noisy,
                noisy_v2=config.noisy_v2,
                image_loss_fn=config.image_loss_fn,
                image_loss_weight=config.image_loss_weight
            )

            resumption_point = None

            last_checkpoint = (checkpointroot
                    / f"stage{i+1}"
                    / f"rnn{stage.res_factor}x_{stage.steps}.pth")

            if i < len(config.stages)-1:
                print(f"Evaluating stage {i}")
                for split in ("train", "val"):
                    eval_stage2(
                        data_json=data_json,
                        savedir=checkpointroot / f"stage{i+1}_artifacts",
                        artifacts=checkpointroot / f"stage{i}_artifacts",
                        res=stage.res_factor,
                        iters=stage.iters,
                        search_range=stage.search_range,
                        patch_factor=stage.patch_factor,
                        checkpoint=checkpointroot
                        / f"stage{i+1}"
                        / f"rnn{stage.res_factor}x_{stage.steps}.pth",
                        diffeomorphic=config.diffeomorphic,
                        split=split,
                    )

if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 3
    main(Path(sys.argv[1]), Path(sys.argv[2]))
