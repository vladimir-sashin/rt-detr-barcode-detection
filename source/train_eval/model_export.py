from argparse import Namespace
from pathlib import Path

from tools.export_onnx import main as rtdetr_to_onnx

from source.train_eval.config.train_eval_cfg import TrainEvalConfig
from source.train_eval.rtdetr_configurator import get_arch_cfg_path
from source.train_eval.training.train import TrainingOutput


def export_onnx(training_output: TrainingOutput, cfg: TrainEvalConfig) -> Path:
    onnx_model_path = training_output.best_checkpoint.parent / 'model.onnx'
    kwargs = {
        'config': get_arch_cfg_path(cfg, training_output.rtdetr_configs_dir),
        'resume': training_output.best_checkpoint,
        'file_name': onnx_model_path,
        'check': True,
        'simplify': False,
    }
    rtdetr_to_onnx(Namespace(**kwargs))
    return onnx_model_path
