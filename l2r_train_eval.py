"""
This script trains the RNN for the Learn2Reg competition

It can also be used to train arbitrary datasets that follow the 
L2R data convention.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from config import TrainConfig
from l2rdata import check_l2r_conformance, get_split_pairs
from train import eval, eval_with_artifacts, train, train_with_artifacts


@dataclass
class RWCNetCheckpoint:
    stage: int
    step: int
    checkpoint: Optional[Path]

def find_last_checkpoint(config: TrainConfig, checkpointroot: Path) -> RWCNetCheckpoint:
    """
    Find the last checkpoint

    NOTE: if the config changes, i.e. previous stage number of steps has changed
    this function won't spot it. Possible to implement this function and retain
    old behaviour with a flag, but its probably better to avoid the situation above
    entirely by specifying a different save path.

    Parameters
    ----------
    config: TrainConfig
    chckpointroot: Path
    """
    num_stages = len(config.stages)
    get_step = lambda x: int(x.split('_')[-1].split('.')[0])
    for stage in range(num_stages, 0, -1):
        stage_folder = checkpointroot / f"stage{stage}"
        if stage_folder.exists():
            checkpoints = [f.name for f in stage_folder.iterdir() if f.name.endswith('.pth')]
            if len(checkpoints) == 0:
                # NOTE: we assume that the checkpoints from the previous stage were generated
                return RWCNetCheckpoint(
                    stage=stage,
                    step=0,
                    checkpoint=None
                )

                # continue
            last_checkpoint = max(checkpoints, key=get_step)
            return RWCNetCheckpoint(
                stage=stage,
                step=get_step(last_checkpoint),
                checkpoint=stage_folder / last_checkpoint
            )
    raise ValueError("Could not find folders for any of the stages")


def main(dataset_json: Path, config_json: Path):
    with open(dataset_json, "r") as f:
        data = json.load(f)

    with open(config_json, "r") as f:
        config_dict = json.load(f)
        config = TrainConfig(**config_dict)

    if config.gpu_num is not None:
        torch.cuda.set_device(config.gpu_num)

    check_l2r_conformance(data)
    root = dataset_json.parent
    split_pairs  = get_split_pairs(data, root, config)

    checkpointroot = Path("checkpoints")
    if config.savedir is not None:
        config.savedir.mkdir(exist_ok=True)
        checkpointroot = config.savedir / Path(f"checkpoints")

    resumption_point: Optional[RWCNetCheckpoint] = None
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
            train(
                data_json=data_json,
                checkpoint_dir=checkpointroot / f"stage{i+1}",
                steps=stage.steps,
                dset_min=config.dset_min,
                dset_max=config.dset_max,
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
                image_loss_fn=stage.image_loss_fn,
                image_loss_weight=stage.image_loss_weight,
                seg_loss_weight=stage.seg_loss_weight,
                lr=stage.lr,
                reg_loss_weight=stage.reg_loss_weight,
                num_workers=config.num_threads,
                switch=config.switch
            )
            resumption_point = None

            last_checkpoint = (checkpointroot
                    / f"stage{i+1}"
                    / f"rnn{stage.res_factor}x_{stage.steps}.pth")

            print(f"Evaluating stage {i}")
            for split in ("train", "val"):
                eval(
                    data_json=data_json,
                    savedir=checkpointroot / f"stage{i+1}_artifacts",
                    res=stage.res_factor,
                    iters=stage.iters,
                    search_range=stage.search_range,
                    dset_min=config.dset_min,
                    dset_max=config.dset_max,
                    patch_factor=stage.patch_factor,
                    checkpoint= last_checkpoint,
                    diffeomorphic=config.diffeomorphic,
                    split=split,
                )
        else:
            print(f"Training stage {i}")
            train_with_artifacts(
                data_json=data_json,
                artifacts=checkpointroot / f"stage{i}_artifacts",
                checkpoint_dir=checkpointroot / f"stage{i+1}",
                dset_min=config.dset_min,
                dset_max=config.dset_max,
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
                image_loss_fn=stage.image_loss_fn,
                image_loss_weight=stage.image_loss_weight,
                seg_loss_weight=stage.seg_loss_weight,
                lr=stage.lr,
                reg_loss_weight=stage.reg_loss_weight,
                switch=config.switch
            )

            resumption_point = None

            last_checkpoint = (checkpointroot
                    / f"stage{i+1}"
                    / f"rnn{stage.res_factor}x_{stage.steps}.pth")

            if i < len(config.stages)-1:
                print(f"Evaluating stage {i}")
                for split in ("train", "val"):
                    eval_with_artifacts(
                        data_json=data_json,
                        savedir=checkpointroot / f"stage{i+1}_artifacts",
                        artifacts=checkpointroot / f"stage{i}_artifacts",
                        res=stage.res_factor,
                        iters=stage.iters,
                        search_range=stage.search_range,
                        dset_min=config.dset_min,
                        dset_max=config.dset_max,
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
