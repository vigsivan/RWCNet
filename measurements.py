from collections import defaultdict
from pathlib import Path
import typer
import json
import numpy as np

app = typer.Typer()

@app.command()
def get_mean(json_file: Path):
    data_dict = defaultdict(list)
    with open(json_file, 'r') as jf:
        data = json.load(jf)
        for dv in data.values():
            for dk, val in dv.items():
                data_dict[dk].append(val)
    for dk, val in data_dict.items():
        print(f"Mean {dk}= ", np.mean(val))

app()
