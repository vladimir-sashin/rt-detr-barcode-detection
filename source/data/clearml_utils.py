from typing import Optional

from clearml import Dataset, Task

from source.data.config import BarcodesDataConfig
from source.data.fetch.logger import LOGGER


class DataManager:
    def __init__(self, project_name: str, task_dataset_name: str):
        self.project_name = project_name
        self.task_dataset_name = task_dataset_name

    def get_ds_if_exists(self, dataset_name: Optional[str] = None, alias: str = 'dataset') -> Optional[Dataset]:
        if dataset_name is None:
            dataset_name = self.task_dataset_name
        existing_ds_names = {ds['name'] for ds in Dataset.list_datasets(dataset_project=self.project_name)}
        if dataset_name in existing_ds_names:
            LOGGER.info("'%s' dataset has been found in ClearML's '%s' project.", dataset_name, self.project_name)
            return Dataset.get(
                dataset_project=self.project_name,
                dataset_name=dataset_name,
                alias=alias,
            )
        return None


def create_dataset_task(cfg: BarcodesDataConfig, dataset_name: str) -> BarcodesDataConfig:
    Task.force_requirements_env_freeze()
    task = Task.init(
        project_name=cfg.clearml_cfg.project_name,
        task_name=dataset_name,
        output_uri=True,  # If `output_uri=True` uses default ClearML output URI
        reuse_last_task_id=False,
    )

    cfg_dump = cfg.model_dump()
    task.connect_configuration(configuration=cfg_dump)
    return BarcodesDataConfig.model_validate(cfg_dump)  # To enable config overriding in ClearML
