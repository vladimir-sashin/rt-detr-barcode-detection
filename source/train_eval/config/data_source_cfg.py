from enum import Enum
from typing import Optional

from pydantic import Field

from source.base_config import BaseValidatedConfig


class StorageEnum(str, Enum):  # noqa: WPS600 str is required here
    clearml = 'clearml'
    local = 'local'


class ClearMLStorageConfig(BaseValidatedConfig):
    project_name: Optional[str] = None
    dataset_version: Optional[str] = None


class DataSourceConfig(BaseValidatedConfig):
    storage: StorageEnum = StorageEnum.local
    dataset_name: str = 'barcodes'
    clearml_storage_cfg: ClearMLStorageConfig = Field(default=ClearMLStorageConfig())
