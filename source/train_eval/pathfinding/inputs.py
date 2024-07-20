from functools import partial
from pathlib import Path

import fiftyone
from clearml import Dataset

from source.base_config import BaseValidatedConfig
from source.data.constants import GT_FIELD_LOADING, SPLITS
from source.data.dir_utils import get_prepared_data_dir
from source.train_eval.config.data_source_cfg import (
    DataSourceConfig,
    StorageEnum,
)
from source.train_eval.config.train_eval_cfg import TrainEvalConfig


class CocoPathConfig(BaseValidatedConfig):
    img_folder: Path
    ann_file: Path

    @property
    def coco_dir(self) -> Path:
        return self.ann_file.parent


class SplitsPathsConfig(BaseValidatedConfig):
    train: CocoPathConfig
    valid: CocoPathConfig
    test: CocoPathConfig


def load_coco_to_fiftyone(split_paths: CocoPathConfig) -> fiftyone.Dataset:
    return fiftyone.Dataset.from_dir(
        dataset_dir=str(split_paths.coco_dir),
        dataset_type=fiftyone.types.COCODetectionDataset,
        label_field=GT_FIELD_LOADING,
    )


def _get_coco_split(coco_dir: Path, split: str) -> CocoPathConfig:
    return CocoPathConfig(
        img_folder=coco_dir / split / 'data',
        ann_file=coco_dir / split / 'labels.json',
    )


def get_coco_splits(coco_dir: Path) -> SplitsPathsConfig:
    get_split_path = partial(_get_coco_split, coco_dir)
    train, valid, test = [get_split_path(split) for split in SPLITS]

    return SplitsPathsConfig(train=train, valid=valid, test=test)


def get_ds_from_cml(data_source_cfg: DataSourceConfig) -> Path:
    return Path(
        Dataset.get(
            dataset_name=data_source_cfg.dataset_name,
            dataset_project=data_source_cfg.clearml_storage_cfg.project_name,
            dataset_version=data_source_cfg.clearml_storage_cfg.dataset_version,
            alias='train_dataset',
        ).get_local_copy(),
    )


def find_dataset(cfg: TrainEvalConfig) -> SplitsPathsConfig:
    if cfg.data_source_cfg.storage == StorageEnum.local:
        dataset_dir = get_prepared_data_dir(cfg.data_source_cfg.dataset_name, tmp=False)
    elif cfg.data_source_cfg.storage == StorageEnum.clearml:
        dataset_dir = get_ds_from_cml(cfg.data_source_cfg)

    return get_coco_splits(dataset_dir)
