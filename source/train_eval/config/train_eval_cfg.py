import os

from pydantic import Field

from source.base_config import BaseValidatedConfig, ConfigYamlMixin
from source.constants import TRAIN_CFG
from source.train_eval.config.clearml_cfg import ClearMLConfig
from source.train_eval.config.data_cfg import DataConfig
from source.train_eval.config.data_source_cfg import DataSourceConfig
from source.train_eval.config.det_model_cfg import DetModelConfig
from source.train_eval.config.runtime_cfg import RuntimeConfig
from source.train_eval.config.training_cfg import TrainingConfig


class TrainEvalConfig(BaseValidatedConfig, ConfigYamlMixin):
    data_source_cfg: DataSourceConfig = Field(DataSourceConfig())
    clearml_cfg: ClearMLConfig = Field(default=ClearMLConfig())
    det_model_cfg: DetModelConfig = Field(default=DetModelConfig())
    training_cfg: TrainingConfig = Field(default=TrainingConfig())
    data_cfg: DataConfig = Field(default=DataConfig())
    runtime_cfg: RuntimeConfig = Field(default=RuntimeConfig())


def get_train_cfg() -> TrainEvalConfig:
    return TrainEvalConfig.from_yaml(os.getenv('TRAIN_CFG_PATH', TRAIN_CFG))
