import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from config import TrainConfig


def check_l2r_conformance(data: Dict):
    """
    Raises a value error if l2r json contains unsupported fields

    Parameters
    ----------
    data: Dict
        dict-ified l2r json
    """
    if data["tensorImageSize"]["0"] != "3D":
        raise ValueError("Only 3D image registration supported")

    if len(set(data["modality"].values())) > 2:
        raise ValueError(
            "This parse script does not support L2R's multimodal JSON format(mostly due to different parse format)."
        )

    image_shape = data["tensorImageShape"]["0"]
    if not all([s % 2 == 0 for s in image_shape]):
        raise ValueError(
            "Expect image dimensions to be at least multiple of 2. Please pad the inputs"
        )


def get_split_pairs(data: Dict, root: Path, config: TrainConfig):
    if data["pairings"] == "paired":
        if not config.random_pairs:
            split_pairs = get_split_pairs_from_paired_dataset(data, root)
        else:
            split_pairs = get_split_pairs_from_unpaired_dataset(data, root)

    elif data["pairings"] == "unpaired":
        split_pairs = get_split_pairs_from_unpaired_dataset(data, root)

    elif data["pairings"] == "hybrid":
        split_pairs = get_split_pairs_from_hybrid_dataset(data, root)
    else:
        raise ValueError(
            f"Expected pairings to be one of paired, unpaired or hybrid but it is {data['pairings']}"
        )
    return split_pairs


def get_split_pairs_from_paired_dataset(data: Dict, root: Path):
    """
    Generates random pairs from an unpaired dataset dictionary (e.g. NLST)
    """
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
    """
    Generates random pairs from an unpaired dataset dictionary (e.g. OASIS)

    From our testing, generating a fixed set of random pairs in the beginning is
    good enough for the OASIS dataset. This is beneficial during training time since
    we can cache flow fields at each stage.
    """
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
                val_images.append(image)
            else:
                print(f"{image} not found")
                break
        else:
            split_data["val"].append(pair_dat)

    train_images = [train_data[k] for k in train_data.keys() if k not in val_images]
    # TODO: expose argument to specify number of pairs per image
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
    for _, di in enumerate(data):
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


