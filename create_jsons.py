import os
import json
import typer
from pathlib import Path

app = typer.Typer()


def give_nickname(name):
    if name == 'HippocampusMR':
        nickname = 'HMR'
    elif name == 'CuRIOUS':
        nickname = 'curious'
    elif name == 'AbdomenMRCT':
        nickname = 'AMRCT'
    elif name == 'AbdomenCTCT':
        nickname = 'ACTCT'
    else:
        nickname = name
    return nickname


@app.command()
def create_json_from_eval(data_directory: Path, json_path: Path, use_nickname: bool = True):

    """
    Creates a json file of the pairs found in the L2R eval config files.

    Compatible with ConvexAdam as of July 6, 2022.

    Parameters
    ----------
    data_directory: Path
        Path to the folder containing all datasets (or parents directory of the dataset directory)
    json_path: Path
        Path to evaluation config (can be found in this repo under jsons/eval_configs)
    use_nickname: bool = True
        True if you use a nickname for the dataset (if you use one to name your folders for conciseness)
        ie Teo uses HMR instead of HippocampusMR

    Returns
    -------

    """

    with open(json_path, 'r') as f:
        data = json.load(f)

    landmarks = ['CuRIOUS', 'LungCT', 'NLST']
    labels = ['AbdomenCTCT', 'AbdomenMRCT', 'HippocampusMR', 'OASIS']

    task_name = data['task_name']
    if use_nickname:
        nickname = give_nickname(task_name)
        data_directory = os.path.join(os.path.sep, data_directory, nickname)
        output_json = nickname + '_eval_pairs_0706.json'
    else:
        data_directory = os.path.join(os.path.sep, data_directory, task_name)
        output_json = task_name + '_eval_pairs_0706.json'

    pairs = data['eval_pairs']
    train_set = []

    for pair in pairs:
        fixed_image = os.path.join(data_directory, pair['fixed'])
        moving_image = os.path.join(data_directory, pair['moving'])

        group = {'fixed_image': fixed_image,
                 'moving_image': moving_image}

        if task_name in landmarks:
            fixed_keypoints = os.path.join(data_directory, pair['fixed'].replace('images','landmarks'))
            moving_keypoints = os.path.join(data_directory, pair['moving'].replace('images', 'landmarks'))
            group['fixed_keypoints'] = fixed_keypoints
            group['moving_keypoints'] = moving_keypoints

        if task_name in labels:
            fixed_segmentation = os.path.join(data_directory, pair['fixed'].replace('images','labels'))
            moving_segmentation = os.path.join(data_directory, pair['moving'].replace('images', 'labels'))
            group['fixed_segmentation'] = fixed_segmentation
            group['moving_segmentation'] = moving_segmentation

        train_set.append(group)

    output_groups = json.dumps({'task_name': task_name,
                                'has_labels': (task_name in 'labels'),
                                'train': train_set})
    with open(output_json, 'w') as outfile:
        outfile.write(output_groups)


app()
