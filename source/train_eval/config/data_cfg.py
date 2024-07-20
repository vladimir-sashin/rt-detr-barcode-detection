from typing import Any, Dict

from pydantic import Field

from source.base_config import BaseValidatedConfig


class DataSplitConfig(BaseValidatedConfig):
    transforms: list[Any] = Field(default=[])  # type: ignore  # Allow explicit `Any`
    shuffle: bool = True
    batch_size: int = 4
    num_workers: int = 4
    drop_last: bool = False


class DataConfig(BaseValidatedConfig):
    num_classes: int = 1  # TODO: calculate from given dataset from local/ClearML
    idx_to_class: Dict[int, str] = {0: 'barcode'}
    train_data: DataSplitConfig = Field(default=DataSplitConfig())
    val_data: DataSplitConfig = Field(default=DataSplitConfig())
    test_data: DataSplitConfig = Field(default=DataSplitConfig())
