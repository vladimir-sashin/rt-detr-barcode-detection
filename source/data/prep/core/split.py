from pathlib import Path
from typing import Annotated

import cv2
import fiftyone
import pandas as pd
from sklearn.model_selection import train_test_split

from source.data.config import DEFAULT_SPLIT_RATIOS, SplitRatios
from source.data.constants import BARCODE_LABEL_NAME, SPLITS
from source.data.prep.logger import LOGGER


def _get_sample(img_meta: pd.Series, input_dir: Path) -> Annotated[list[fiftyone.Sample], 1]:
    filepath = str(Path(input_dir) / Path(img_meta.filename))
    height, width = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE).shape
    bbox = [
        img_meta.x_from / width,
        img_meta.y_from / height,
        img_meta.width / width,
        img_meta.height / height,
    ]
    detection = fiftyone.Detection(
        label=BARCODE_LABEL_NAME,
        bounding_box=bbox,
    )

    sample = fiftyone.Sample(filepath=filepath, tags=[img_meta.split])
    sample['ground_truth'] = fiftyone.Detections(detections=[detection])
    # Pandas calls 'iter()' on value returned by 'DataFrame.apply()' method, which leads 'fiftyone.Sample.__iter__()'
    # to raise ValueError that is not handled by pandas, so we need to work around it
    return [sample]


def get_samples(meta_df: pd.DataFrame, input_dir: Path) -> pd.Series:
    samples = meta_df.apply(_get_sample, input_dir=input_dir, axis=1, result_type='expand')
    return samples.squeeze()


def read_split_df(tsv_path: Path, ratios: SplitRatios, random_state: int) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep='\t')

    train, remaining = train_test_split(df, train_size=ratios.train, random_state=random_state)
    valid, test = train_test_split(
        remaining,
        test_size=ratios.test / (ratios.test + ratios.valid),
        random_state=random_state,
    )

    train['split'] = SPLITS.train
    valid['split'] = SPLITS.valid
    test['split'] = SPLITS.test

    return pd.concat([train, valid, test], axis=0)


def _get_tsv_path(raw_dataset_path: Path) -> Path:
    try:
        tsv_path = next(raw_dataset_path.rglob('*.tsv'))
    except StopIteration:
        raise ValueError(f'annotations.tsv file is not found in {raw_dataset_path}.')
    return tsv_path


def split_raw_dataset(
    raw_dataset_path: Path,
    ratios: SplitRatios = DEFAULT_SPLIT_RATIOS,
    random_state: int = 42,
) -> fiftyone.Dataset:
    tsv_path = _get_tsv_path(raw_dataset_path)
    df = read_split_df(tsv_path, ratios, random_state)

    samples = get_samples(df, tsv_path.parent)
    dataset = fiftyone.Dataset()
    dataset.add_samples(samples)
    LOGGER.info('Raw zipped barcodes dataset is successfully split into train/val/test sets')

    return dataset
