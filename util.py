import json
from pathlib import Path
from typing import Optional

import nibabel as nib
import typer

app = typer.Typer()

@app.command()
def oasis(dataset_json: Path, dataset_path: Path, output_json: Path):
    with open(dataset_json, "r") as f:
        l2r_data = json.load(f)

    training_data = {
        Path(subject["image"]).name: subject for subject in l2r_data["training"]
    }

    images = l2r_data["training"]
    val_pairs = l2r_data["registration_val"]
    val_images = set(p["fixed"] for p in val_pairs).union(p["moving"] for p in val_pairs)

    train_data = []
    val_data = []
    labels = list(range(0,36))
    mask_dir: Optional[Path] = None
    label_dir: Optional[Path] = None
    for image in images:
        data_dict = val_data if image["image"] in val_images else train_data
        labels_nib = nib.load(dataset_path / image["label"]).get_fdata()
        if image["image"] in val_images:
            continue
        if mask_dir is None:
            mask_dir = dataset_path / Path(image["mask"]).parent
        if label_dir is None:
            label_dir = dataset_path / Path(image["label"]).parent
        train_data.append(
            {
                "image": str(dataset_path / image["image"]),
                "label": str(dataset_path / image["label"]),
                "mask": str(dataset_path / image["mask"]),
            }
        )

    for val_pair in val_pairs:
        assert mask_dir is not None and label_dir is not None
        data = {
            "fixed_image": str(dataset_path / val_pair["fixed"]),
            "moving_image": str(dataset_path / val_pair["moving"]),
            "fixed_segmentation": str(label_dir / Path(val_pair["fixed"]).name),
            "moving_segmentation": str(label_dir / Path(val_pair["moving"]).name),
            "fixed_mask": str(mask_dir / Path(val_pair["fixed"]).name),
            "moving_mask": str(mask_dir / Path(val_pair["moving"]).name),
        }
        for p in data.values():
            assert Path(p).exists()
        val_data.append(data)


    with open(output_json, "w") as f:
        json.dump({"labels": labels, "train": train_data, "val": val_data}, f)

@app.command()
def convert_nlst_json(dataset_json: Path, dataset_path: Path, output_json: Path):
    with open(dataset_json, "r") as f:
        l2r_data = json.load(f)

    training_data = {
        Path(subject["image"]).name: subject for subject in l2r_data["training"]
    }

    pairs = l2r_data["training_paired_images"]
    val_pairs = l2r_data["registration_val"]
    val_images_fixed = set(p["fixed"] for p in val_pairs)
    train_data = []
    val_data = []
    for pair in pairs:
        fixed, moving = Path(pair["fixed"]), Path(pair["moving"])
        fixed_data, moving_data = training_data[fixed.name], training_data[moving.name]
        fixed_mask, moving_mask = fixed_data["mask"], moving_data["mask"]
        fixed_keypoints, moving_keypoints = (
            fixed_data["keypoints"],
            moving_data["keypoints"],
        )
        fixed_keypoints, moving_keypoints = [
            kp.replace(".nii.gz", ".csv")  # lol
            for kp in (fixed_keypoints, moving_keypoints)
        ]
        data_dict = val_data if pair["fixed"] in val_images_fixed else train_data
        data_dict.append(
            {
                "fixed_image": str(dataset_path / fixed),
                "moving_image": str(dataset_path / moving),
                "fixed_segmentation": str(dataset_path / fixed_mask),
                "moving_segmentation": str(dataset_path / moving_mask),
                "fixed_keypoints": str(dataset_path / fixed_keypoints),
                "moving_keypoints": str(dataset_path / moving_keypoints),
            }
        )
    with open(output_json, "w") as f:
        json.dump({"labels": [1], "train": train_data, "val": val_data}, f)



app()
