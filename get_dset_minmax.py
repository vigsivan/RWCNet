from pathlib import Path
import typer
import nibabel as nib
import numpy as np
import multiprocessing

app = typer.Typer()

def find_min_max(image_path: Path):
    image_arr = nib.load(image_path).get_fdata()
    return np.min(image_arr).item(), np.max(image_arr).item()


@app.command()
def main(folder: Path, j: int=4):
    files = list(folder.iterdir())
    with multiprocessing.Pool(j) as p:
        min_maxes = p.map(find_min_max, files)
    min_val = min([i[0] for i in min_maxes])
    max_val = max([i[1] for i in min_maxes])
    print(f"Minimum: {min_val}\tMaximum: {max_val}")


app()
