from __future__ import annotations
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field, PositiveInt, NonNegativeInt
import yaml

class OptimizerConfig(BaseModel):
    lr: float = 5e-4
    weight_decay: float = 0.0
    betas: List[float] = [0.9, 0.999]
    eps: float = 1e-8
    momentum: Optional[float] = None 


class SchedulerConfig(BaseModel):
    warmup_steps: NonNegativeInt = 0
    initial_lr: float = 3e-5
    min_lr: float = 1e-6


class EvalConfig(BaseModel):
    eval_freq: PositiveInt = 1000
    eval_iter: PositiveInt = 50


class CheckpointConfig(BaseModel):
    checkpoint_dir: str = "checkpoints"
    keep_last_n_epochs: NonNegativeInt = 3   # stricter than Optional[int]
    best_model_filename: str = "best_model.pth"
    best_checkpoint_filename: str = "best_checkpoint.pth"
    final_checkpoint_filename: str = "final_checkpoint.pth"
    epoch_pattern: str = "epoch_{:04d}.pth"


class SamplingConfig(BaseModel):
    enabled: bool = True
    start_context: str = "नेपाल सुन्दर देश हो"
    max_new_tokens: PositiveInt = 50


class TrainingConfig(BaseModel):
    n_epochs: PositiveInt = 1
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)


def load_training_config(path: str | Path) -> TrainingConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    payload = raw.get("training", raw)
    return TrainingConfig.model_validate(payload)