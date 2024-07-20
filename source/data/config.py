import os
from dataclasses import dataclass

from pydantic import Field

from source.base_config import BaseValidatedConfig, ConfigYamlMixin
from source.constants import DATA_CFG
from source.data.constants import DEFAULT_RAW_URL, DEFAULT_SPLIT_RATIO_VALS
from source.train_eval.config.data_source_cfg import StorageEnum


@dataclass
class SplitRatios:
    train: float
    valid: float
    test: float

    def __post_init__(self) -> None:
        eps = 1e-6
        total = self.train + self.valid + self.test
        if abs(1 - total) > eps:
            raise ValueError(f'Split ratios should sum up to 1, got {total}')


DEFAULT_SPLIT_RATIOS = SplitRatios(*DEFAULT_SPLIT_RATIO_VALS)


class SplitConfig(BaseValidatedConfig):
    seed: int = 42
    split_ratios: SplitRatios = DEFAULT_SPLIT_RATIOS


class ClearMLConfig(BaseValidatedConfig):
    project_name: str = 'Barcodes detection'
    keep_local_copy: bool = False


class BarcodesDataConfig(BaseValidatedConfig, ConfigYamlMixin):
    dataset_name: str = 'barcodes'
    url: str = DEFAULT_RAW_URL
    split_cfg: SplitConfig = Field(default=SplitConfig())
    storage: StorageEnum = StorageEnum.local
    clearml_cfg: ClearMLConfig = Field(default=ClearMLConfig())


def get_data_config() -> BarcodesDataConfig:
    return BarcodesDataConfig.from_yaml(os.getenv('DATA_CFG_PATH', DATA_CFG))
