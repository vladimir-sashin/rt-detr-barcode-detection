from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar

from source.constants import OUTPUT_DIR
from source.train_eval.config.train_eval_cfg import TrainEvalConfig

InputParam = ParamSpec('InputParam')
RetType = TypeVar('RetType')


def _get_datetime() -> str:
    return f'run_{datetime.now().strftime("%d-%m-%y_%H-%M-%S")}'


def execute_once(func: Callable[InputParam, RetType]) -> Callable[InputParam, RetType]:
    @wraps(func)
    def wrapper(*args: InputParam.args, **kwargs: InputParam.kwargs) -> RetType:
        if wrapper.cached_output is None:  # type: ignore
            func_output = func(*args, **kwargs)
            wrapper.cached_output = func_output  # type: ignore
        return wrapper.cached_output  # type: ignore

    wrapper.cached_output = None  # type: ignore

    return wrapper


@execute_once
def get_output_dir(cfg: TrainEvalConfig) -> Path:
    if cfg.clearml_cfg.track_in_clearml is True:
        output_dir = (
            OUTPUT_DIR / 'clearml_on' / cfg.clearml_cfg.project_name / cfg.clearml_cfg.experiment_name / _get_datetime()
        )
    else:
        output_dir = OUTPUT_DIR / 'clearml_off' / cfg.data_source_cfg.dataset_name / _get_datetime()

    output_dir.mkdir(exist_ok=True, parents=True)
    return output_dir
