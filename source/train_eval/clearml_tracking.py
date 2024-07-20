from pathlib import Path
from typing import List, Optional, Tuple

from clearml import OutputModel, Task

from source.train_eval.config.train_eval_cfg import TrainEvalConfig
from source.train_eval.evaluation.results_export import CompleteEvalResults
from source.train_eval.logger import LOGGER


def upload_model(task: Task, checkpoint_path: Path, model_name: str, tags: Optional[List[str]] = None) -> None:
    # TODO: log labels:id map to ClearML model (extract it from config connected to the task)
    LOGGER.info('Uploading best %s checkpoint to ClearML...', checkpoint_path.suffix)
    output_model = OutputModel(task=task, name=model_name, tags=tags)
    output_model.update_weights(weights_filename=str(checkpoint_path), auto_delete_file=False)


def create_cml_task(cfg: TrainEvalConfig) -> Task:
    cml_cfg = cfg.clearml_cfg
    Task.force_requirements_env_freeze()
    return Task.init(
        project_name=cml_cfg.project_name,
        task_name=cml_cfg.experiment_name,
        output_uri=True,  # If `output_uri=True` uses default ClearML output URI
        auto_connect_frameworks={
            'pytorch': False,
            'matplotlib': False,
        },
        reuse_last_task_id=False,
    )


def sync_cfg_with_cml(cfg: TrainEvalConfig, task: Task) -> TrainEvalConfig:
    cfg_dump = cfg.model_dump()
    task.connect_configuration(configuration=cfg_dump)
    return TrainEvalConfig.model_validate(cfg_dump)


def setup_clearml(cfg: TrainEvalConfig) -> Tuple[TrainEvalConfig, Optional[Task]]:
    if cfg.clearml_cfg.track_in_clearml is True:
        task = create_cml_task(cfg)
        cfg = sync_cfg_with_cml(cfg, task)
        return cfg, task
    return cfg, None


def track_artifacts(
    task: Optional[Task],
    best_pt_checkpoint: Path,
    onnx_model: Path,
    val_eval_res: CompleteEvalResults,
    test_eval_res: CompleteEvalResults,
) -> None:
    # If ClearML tracking is disabled
    if task is None:
        return

    cml_logger = task.get_logger()

    val_eval_res.track_in_clearml(cml_logger, 'val')
    test_eval_res.track_in_clearml(cml_logger, 'test')
    upload_model(task, best_pt_checkpoint, model_name='best_pth_checkpoint', tags=['torch', 'best_checkpoint'])
    upload_model(task, onnx_model, model_name='onnx_model', tags=['onnx', 'best_checkpoint'])
    LOGGER.info(
        'Visualizations of predicts, detailed val/test evaluations metrics and plots has been '
        + 'successfully uploaded to CLearML!',
    )
