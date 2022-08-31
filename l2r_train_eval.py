from collections import defaultdict
import json
from typing import Dict, Optional, List
import sys
import random
from pathlib import Path
import nibabel as nib
import torch
import typer
import numpy as np
import os
from tqdm import tqdm

from common import MINDSSC
from train import train_stage1, train_stage2, eval_stage1, eval_stage2, eval_stage3

app = typer.Typer()


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

    for split_p in ("training", "test"):
        for dat in data[f"{split_p}_paired_images"]:
            split = "val" if dat["fixed"] in val_fixeds else "train"
            pair_dat = {}
            for lab in ("fixed", "moving"):
                image = dat[lab]
                if image in train_data:
                    all_dat_for_image = train_data[image]
                    pair_dat.update(
                        {f"{lab}_{k}": v for k, v in all_dat_for_image.items()}
                    )
                else:
                    print(f"{image} not found")
                    break
            else:
                split_data[split].append(pair_dat)

    return split_data


def get_split_pairs_from_unpaired_dataset(data: Dict, root: Path):
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
            if image in train_data:
                all_dat_for_image = train_data[image]
                pair_dat.update({f"{lab}_{k}": v for k, v in all_dat_for_image.items()})
            else:
                print(f"{image} not found")
                break
        else:
            split_data["test"].append(pair_dat)

    return split_data


def get_pairs_from_list(
    data: List[Dict[str, Path]], pairs_per_image: int = 5, root: Optional[Path] = None
) -> List[Dict[str, Path]]:
    return get_random_pairs_from_list(data, pairs_per_image)


def get_random_pairs_from_list(
    data: List[Dict[str, Path]], pairs_per_image: int = 5
) -> List[Dict[str, Path]]:
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


@app.command()
def main(dataset_json: Path, output_recipe: Optional[Path] = None):
    with open(dataset_json, "r") as f:
        data = json.load(f)

    root = dataset_json.parent
    metadata = {}
    training_recipe = {}

    metadata["name"] = data["name"]

    if data["tensorImageSize"]["0"] != "3D":
        raise ValueError("Only 3D image registration supported")

    if len(set(data["modality"].values())) > 2:
        raise ValueError(
            "This parse script does not support multimodal (mostly due to different parse format)."
        )

    if data["pairings"] == "paired":
        split_pairs = get_split_pairs_from_paired_dataset(data, root)

    elif data["pairings"] == "unpaired":
        split_pairs = get_split_pairs_from_unpaired_dataset(data, root)

    elif data["pairings"] == "hybrid":
        split_pairs = get_split_pairs_from_hybrid_dataset(data, root)
    else:
        ValueError(
            f"Expected pairings to be one of paired, unpaired or hybrid but it is {data['pairings']}"
        )

    image_shape = data["tensorImageShape"]["0"]
    if not all([s % 2 == 0 for s in image_shape]):
        raise ValueError(
            "Expect image dimensions to be at least multiple of 2. Please pad the inputs"
        )

    data_json = Path(f"{data['name']}.json")

    with open(data_json, "w") as f:
        json.dump(split_pairs, f)

    checkpointroot = Path(f"{data['name']}_checkpoints")
    checkpointroot.mkdir(exist_ok=True)
    num_threads = os.cpu_count()
    if num_threads is None:
        num_threads = 4

    if all([s % 4 == 0 for s in image_shape]):
        print("Running 3-stage training")
        print("Training stage 1")
        # train_stage1(
        #     data_json=data_json,
        #     checkpoint_dir=checkpointroot / "stage1",
        #     res=4,
        #     patch_factor=4,
        #     steps=15000,
        #     num_workers=num_threads,
        #     diffeomorphic=True,
        # )

        # print("Generating stage 1 artifacts")
        # eval_stage1(
        #     data_json=data_json,
        #     savedir=checkpointroot / "stage1_artifacts",
        #     res=4,
        #     patch_factor=4,
        #     checkpoint=checkpointroot / "stage1" / "rnn4x_15000.pth",
        #     diffeomorphic=True,
        #     split="val"
        # )
        # eval_stage1(
        #     data_json=data_json,
        #     savedir=checkpointroot / "stage1_artifacts",
        #     res=4,
        #     patch_factor=4,
        #     checkpoint=checkpointroot / "stage1" / "rnn4x_15000.pth",
        #     diffeomorphic=True,
        #     split="train"
        # )
        # eval_stage1(
        #     data_json=data_json,
        #     savedir=checkpointroot / "stage1_artifacts",
        #     res=4,
        #     patch_factor=4,
        #     checkpoint=checkpointroot / "stage1" / "rnn4x_15000.pth",
        #     diffeomorphic=True,
        #     split="test"
        # )

        # print("Training stage 2")
        # train_stage2(
        #     data_json=data_json,
        #     checkpoint_dir=checkpointroot / "stage2",
        #     artifacts=checkpointroot / "stage1_artifacts",
        #     res=2,
        #     patch_factor=4,
        #     steps=25000,
        #     num_workers=num_threads,
        #     diffeomorphic=True,
        #     start=checkpointroot / "stage2" / "rnn2x_15000.pth",
        #     iters=12,
        #     search_range=3
        # )
        #
        # print("Generating stage 2 artifacts")
        # eval_stage2(
        #     data_json=data_json,
        #     savedir=checkpointroot / "stage2_artifacts",
        #     artifacts=checkpointroot / "stage1_artifacts",
        #     res=2,
        #     patch_factor=4,
        #     checkpoint=checkpointroot / "stage2" / "rnn2x_20000.pth",
        #     diffeomorphic=True,
        #     split="val"
        # )
        # eval_stage2(
        #     data_json=data_json,
        #     savedir=checkpointroot / "stage2_artifacts",
        #     artifacts=checkpointroot / "stage1_artifacts",
        #     res=2,
        #     patch_factor=4,
        #     checkpoint=checkpointroot / "stage2" / "rnn2x_15000.pth",
        #     diffeomorphic=True,
        #     split="train"
        # )
        # eval_stage2(
        #     data_json=data_json,
        #     savedir=checkpointroot / "stage2_artifacts",
        #     artifacts=checkpointroot / "stage1_artifacts",
        #     res=2,
        #     patch_factor=4,
        #     checkpoint=checkpointroot / "stage2" / "rnn2x_15000.pth",
        #     diffeomorphic=True,
        #     split="test"
        # )

        print("Training stage 3")
        train_stage2(
            data_json=data_json,
            checkpoint_dir=checkpointroot / "stage3",
            artifacts=checkpointroot / "stage2_artifacts",
            res=1,
            patch_factor=2,
            steps=10000,
            num_workers=num_threads,
            diffeomorphic=True,
            iters=4,
            search_range=2
        )

    else:
        print("Running 2-stage training")
        print("Training stage 1")
        train_stage1(
            data_json=data_json,
            checkpoint_dir=checkpointroot / "stage1",
            res=2,
            patch_factor=2,
            steps=15000,
            num_workers=num_threads,
            diffeomorphic=True,
        )
        print("Generating stage 1 artifacts")
        eval_stage1(
            data_json=data_json,
            savedir=checkpointroot / "stage1_artifacts",
            res=2,
            patch_factor=2,
            checkpoint=checkpointroot / "stage1" / "rnn2x_15000.pth",
            diffeomorphic=True,
            split="val"
        )
        eval_stage1(
            data_json=data_json,
            savedir=checkpointroot / "stage1_artifacts",
            res=2,
            patch_factor=2,
            checkpoint=checkpointroot / "stage1" / "rnn2x_15000.pth",
            diffeomorphic=True,
            split="train"
        )
        eval_stage1(
            data_json=data_json,
            savedir=checkpointroot / "stage1_artifacts",
            res=2,
            patch_factor=2,
            checkpoint=checkpointroot / "stage1" / "rnn2x_15000.pth",
            diffeomorphic=True,
            split="test"
        )
        print("Training stage 2")
        train_stage2(
            data_json=data_json,
            checkpoint_dir=checkpointroot / "stage2",
            artifacts=checkpointroot / "stage1_artifacts",
            res=1,
            patch_factor=2,
            steps=10000,
            num_workers=num_threads,
            diffeomorphic=True,
            iters=4,
            search_range=2
        )

app()
