from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, validator

class TrainStageConfig(BaseModel):
    res_factor: int
    patch_factor: int
    steps: int
    stage_number: Optional[int]=None
    starting_checkpont: Optional[Path]=None
    iters: int = 12
    search_range: int = 3
    start_from_last: bool =False
    save_freq: int=100
    log_freq: int=10
    val_freq: int=100

class TrainConfig(BaseModel):
    stages: List[TrainStageConfig]
    cache_file: Path
    savedir: Optional[Path]=None
    num_threads: int=4
    device: str = "cuda"
    diffeomorphic: bool=True

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
