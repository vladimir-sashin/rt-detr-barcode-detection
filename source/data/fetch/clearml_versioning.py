from pathlib import Path

from clearml import Dataset

from source.data.clearml_utils import DataManager
from source.data.fetch.logger import LOGGER


class RawDataManager(DataManager):
    def upload_raw_dataset(self, dataset_path: Path) -> None:
        if self.get_ds_if_exists(alias='raw_dataset'):
            LOGGER.info('Skipped uploading of dataset to ClearML.')
            return
        self._upload_raw_ds(dataset_path)
        LOGGER.info('Dataset is successfully uploaded to ClearML.')

    def _upload_raw_ds(self, dataset_path: Path) -> Dataset:
        # TODO: add possibility to pass dataset description
        dataset = Dataset.create(
            dataset_project=self.project_name,
            use_current_task=True,
        )
        dataset.add_files(dataset_path)
        dataset.tags = ['raw']
        dataset.finalize(auto_upload=True)
        return dataset
