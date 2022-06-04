from pathlib import Path
import json
import typer

app = typer.Typer()


@app.command()
def convert_nlst_json(dataset_json: Path, dataset_path: Path, output_json: Path):
    with open(dataset_json, "r") as f:
        l2r_data = json.load(f)

    training_data = {
        Path(subject["image"]).name: subject for subject in l2r_data["training"]
    }

    pairs = l2r_data["training_paired_images"]
    data = []
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
        data.append(
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
        json.dump({"data": data}, f)


app()
