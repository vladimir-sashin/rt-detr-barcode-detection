import shutil
from pathlib import Path

import fiftyone

from source.constants import PROJECT_ROOT
from source.data.constants import GT_FIELD_LOADING, SPLITS
from source.data.prep.logger import LOGGER


def splits_to_coco(dataset_splits: fiftyone.Dataset, export_dir: Path) -> None:
    LOGGER.info('Exporting barcodes dataset splits in COCO format to %s...', export_dir.relative_to(PROJECT_ROOT))
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)

    for split in SPLITS:
        dataset_splits.match_tags(split).export(
            export_dir=str(export_dir / split),
            dataset_type=fiftyone.types.COCODetectionDataset,
            label_field=GT_FIELD_LOADING,
        )
    LOGGER.info('Barcodes dataset splits are successfully exported to %s.', export_dir)
