from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, validator

class TrainStageConfig(BaseModel):
    iters: int = 12
    search_range: int = 3
    res_factor: int
    patch_factor: int
    steps: int
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
    savedir: Path
    num_threads: int=4
    device: str = "cuda"
    diffeomorphic: bool=True
    overwrite: bool = False
    gpu_num: Optional[int]=None
    use_best_validation_checkpoint: bool=True
    dset_min: float=-4e-3
    dset_max: float=16e3

    @validator("stages")
    def validate_stages(cls, val):
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
    save_path: Path
    dset_min: float
    dset_max: float
    instance_opt_res: int=2
    eval_at_each_stage: bool=False
    split: str = "test"
    device: str = "cuda"

    @validator("stages")
    def validate_stages(cls, val):
        if len(val) == 0:
            raise ValueError("Expected at least one stage")
        return val
