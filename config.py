from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, validator
from networks import SomeNet, SomeNetNoCorr, SomeNetNoisy, SomeNetNoisyv2
import torch

class TrainStageConfig(BaseModel):
    res_factor: int
    patch_factor: int
    steps: int
    iters: int = 12
    search_range: int = 3
    save_freq: int=100
    seg_loss_weight: float=1
    log_freq: int=10
    val_freq: int=100
    image_loss_fn: str="mse"
    image_loss_weight: float=10.
    reg_loss_weight: float=.1
    lr: float=3e-4

    @validator("image_loss_fn")
    def validate_image_loss(cls, val):
        losses = ("mi", "mse", "ncc")
        if val not in losses:
            raise ValueError(f"Expected {val} to be in {losses}")
        return val

class TrainConfig(BaseModel):
    stages: List[TrainStageConfig]
    cache_file: Path
    savedir: Path
    num_threads: int=4
    device: str = "cuda"
    diffeomorphic: bool=True
    overwrite: bool = False
    gpu_num: Optional[int]=None
    noisy: bool=False
    noisy_v2: bool=False #FIXME: refactor when you figure out which noisy is better
    use_best_validation_checkpoint: bool=True
    dset_min: float=-4e-3
    dset_max: float=16e3

    @validator("stages")
    def validate_cache_file(cls, val):
        if len(val) == 0:
            raise ValueError("Expected at least one stage")
        return val

class EvalStageConfig(BaseModel):
    res_factor: int
    patch_factor: int
    checkpoint: Path
    iters: int = 12
    search_range: int = 3
    diffeomorphic: bool = True

class EvalConfig(BaseModel):
    stages: List[EvalStageConfig]
    cache_file: Path
    save_path: Path
    instance_opt_res: int=2
    eval_at_each_stage: bool=False
    split: str = "test"
    device: str = "cuda"

    @validator("stages")
    def validate_cache_file(cls, val):
        if len(val) == 0:
            raise ValueError("Expected at least one stage")
        return val
