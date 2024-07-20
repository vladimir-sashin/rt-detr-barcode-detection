from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import Field, field_validator

from source.base_config import BaseValidatedConfig
from source.constants import PROJECT_ROOT
from source.train_eval.constants import RtdetrOptims, RtdetrSchedulers


class EmaConfig(BaseValidatedConfig):
    use_ema: bool = True
    kwargs: Dict[str, Any] = Field(default_factory=dict)  # type: ignore  # Allow explicit `Any`


class OptimizerConfig(BaseValidatedConfig):
    target_class: RtdetrOptims = RtdetrOptims.adam_w
    kwargs: Dict[str, Any] = Field(default_factory=dict)  # type: ignore  # Allow explicit `Any`


class LRSchedulerConfig(BaseValidatedConfig):
    target_class: RtdetrSchedulers = RtdetrSchedulers.multi_step
    kwargs: Dict[str, Any] = Field(default_factory=dict)  # type: ignore  # Allow explicit `Any`


class TrainingConfig(BaseValidatedConfig):
    epochs: int = 100
    resume_checkpoint: Optional[Path] = None
    seed: int = 42
    ema: EmaConfig = Field(default=EmaConfig())
    clip_max_norm: float = 0.1
    find_unused_parameters: bool = True
    optimizer: OptimizerConfig = Field(default=OptimizerConfig())
    lr_scheduler: LRSchedulerConfig = Field(default=LRSchedulerConfig())

    @field_validator('resume_checkpoint')
    @classmethod
    def make_absolute_path(cls, path: Optional[str]) -> Optional[Path]:
        return _get_abs_path(path)


def _get_abs_path(relative_path: Optional[str]) -> Optional[Path]:
    if relative_path is None:
        return relative_path
    return PROJECT_ROOT / relative_path
