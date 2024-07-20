from pathlib import Path
from typing import Optional, Sequence

from clearml import Dataset

from source.data.clearml_utils import DataManager
from source.data.dir_utils import get_raw_ds_name
from source.data.prep.logger import LOGGER


class CMLDatasetNotFound(ValueError):
    """Dataset not found in ClearML."""


class PreprocessDataManager(DataManager):
    def __init__(self, project_name: str, task_dataset_name: str):
        super().__init__(project_name, task_dataset_name)
        self.raw_dataset: Dataset = self._check_raw_ds_exists()

    def upload_processed_ds(
        self,
        processed_dir: Path,
        tags: Sequence[str] = ('preprocessed',),
    ) -> Dataset:
        processed_ds = Dataset.create(
            use_current_task=True,
            dataset_project=self.project_name,
            parent_datasets=[self.raw_dataset],
            dataset_tags=tags,
        )
        processed_ds.sync_folder(local_path=processed_dir)
        processed_ds.finalize(auto_upload=True)
        return processed_ds

    def _check_raw_ds_exists(self) -> Dataset:
        raw_ds_name = get_raw_ds_name(self.task_dataset_name)
        if (raw_dataset := self.get_ds_if_exists(dataset_name=raw_ds_name, alias='raw_dataset')) is None:
            raise CMLDatasetNotFound(
                f"Dataset with name '{raw_ds_name}' doesn't exist in ClearML project '{self.project_name}'.",
            )
        return raw_dataset


def get_ds_local_copy(dataset: Dataset) -> Path:
    return Path(dataset.get_local_copy())


def create_data_manager(project_name: str, dataset_name: str, raw_dataset_name: str) -> Optional[PreprocessDataManager]:
    try:
        data_manager = PreprocessDataManager(
            project_name=project_name,
            task_dataset_name=dataset_name,
        )
    except CMLDatasetNotFound as exp:
        LOGGER.info('%s', repr(exp))
        LOGGER.info(
            'To use ClearML dataset tracking feature, please upload raw version of the dataset with the "%s" '
            + 'name to the "%s" project and try again, or disable ClearML dataset tracking in the data config to '
            + 'use local data: set "clearml_cfg.upload_to_clearml: False".',
            raw_dataset_name,
            project_name,
        )
        LOGGER.warning('DATASET PROCESSING IS ABORTED')
        return None
    return data_manager
