import os
from pathlib import Path

import fiftyone


def main(coco_dir: Path) -> None:
    dataset = fiftyone.Dataset.from_dir(
        dataset_dir=coco_dir,
        dataset_type=fiftyone.types.COCODetectionDataset,
        label_field='ground_truth',
    )
    session = fiftyone.launch_app(dataset)
    session.wait()


if __name__ == '__main__':
    coco_dir = os.getenv('COCO_DIR')
    main(coco_dir)  # type: ignore
