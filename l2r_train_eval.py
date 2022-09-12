from collections import defaultdict
import json
from typing import Dict, Optional, List
import sys
import random
from pathlib import Path
from config import TrainConfig

from train import train_stage1, train_stage2, eval_stage1, eval_stage2

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


def main(dataset_json: Path, config_json: Path):
    with open(dataset_json, "r") as f:
        data = json.load(f)

    with open(config_json, "r") as f:
        config_dict = json.load(f)
        config = TrainConfig(**config_dict)

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

    checkpointroot.mkdir(exist_ok=True)

    data_json = checkpointroot / f"data.json"

    with open(data_json, "w") as f:
        json.dump(split_pairs, f)

    last_checkpoint = None
    for i, stage in enumerate(config.stages):
        if i == 0:
            train_stage1(
                data_json=data_json,
                checkpoint_dir=checkpointroot / f"stage{i+1}",
                steps=stage.steps,
                res=stage.res_factor,
                patch_factor=stage.patch_factor,
                iters=stage.iters,
                search_range=stage.search_range,
                diffeomorphic=stage.diffeomorphic,
                save_freq=stage.save_freq,
                log_freq=stage.log_freq,
                val_freq=stage.val_freq
            )

            last_checkpoint = (checkpointroot
                    / f"stage{i+1}"
                    / f"rnn{stage.res_factor}x_{stage.steps}.pth")

            for split in ("train", "val"):
                eval_stage1(
                    data_json=data_json,
                    savedir=checkpointroot / f"stage{i+1}_artifacts",
                    res=stage.res_factor,
                    iters=stage.iters,
                    search_range=stage.search_range,
                    patch_factor=stage.patch_factor,
                    checkpoint= last_checkpoint,
                    diffeomorphic=stage.diffeomorphic,
                    split=split,
                )
        else:
            train_stage2(
                data_json=data_json,
                artifacts=checkpointroot / f"stage{i}_artifacts",
                checkpoint_dir=checkpointroot / f"stage{i+1}",
                steps=stage.steps,
                res=stage.res_factor,
                patch_factor=stage.patch_factor,
                iters=stage.iters,
                search_range=stage.search_range,
                diffeomorphic=stage.diffeomorphic,
                start=( None if not stage.start_from_last
                        else last_checkpoint),
                save_freq=stage.save_freq,
                log_freq=stage.log_freq,
                val_freq=stage.val_freq

            )

            last_checkpoint = (checkpointroot
                    / f"stage{i+1}"
                    / f"rnn{stage.res_factor}x_{stage.steps}.pth")

            if i < len(config.stages)-1:
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
                        diffeomorphic=stage.diffeomorphic,
                        split=split,
                    )

if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 3
    main(Path(sys.argv[1]), Path(sys.argv[2]))
