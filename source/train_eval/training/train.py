from argparse import Namespace
from pathlib import Path
from typing import NamedTuple

from source.train_eval.config.train_eval_cfg import TrainEvalConfig
from source.train_eval.logger import LOGGER
from source.train_eval.rtdetr_configurator import (
    generate_rtdetr_configs,
    get_arch_cfg_path,
)
from source.train_eval.training.pretrain_loading import load_pretrained
from third_party.rtdetr_pytorch.tools.train import main as rtdetr_main


class TrainingOutput(NamedTuple):
    best_checkpoint: Path
    rtdetr_configs_dir: Path


def run_training(cfg: TrainEvalConfig) -> TrainingOutput:
    # Generate RTDETR configs based on our train.yaml cfg
    rtdetr_configs_dir = generate_rtdetr_configs(cfg)
    # Run RTDETR training
    best_checkpoint = _run_training(cfg, rtdetr_configs_dir)
    return TrainingOutput(best_checkpoint, rtdetr_configs_dir)


def _run_training(cfg: TrainEvalConfig, rtdetr_configs_dir: Path) -> Path:
    # Load pretrain
    checkpoint_path = str(load_pretrained(cfg.det_model_cfg))
    # Run RTDETR training
    kwargs = {
        'config': get_arch_cfg_path(cfg, rtdetr_configs_dir),
        'resume': cfg.training_cfg.resume_checkpoint,
        'amp': cfg.runtime_cfg.use_amp,
        'tuning': checkpoint_path,
        'seed': cfg.training_cfg.seed,
        'test_only': False,
    }
    best_checkpoint = rtdetr_main(Namespace(**kwargs))
    LOGGER.info('Training finished, best checkpoint is saved at %s', best_checkpoint)
    return best_checkpoint
