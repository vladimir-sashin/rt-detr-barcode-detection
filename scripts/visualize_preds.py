import os
from typing import Optional

import fiftyone


def create_fiftyone_dataset(
    images_dir: str,
    preds_coco_labels: str,
    gt_coco_labels: Optional[str] = None,
) -> fiftyone.Dataset:
    dataset = fiftyone.Dataset.from_dir(
        data_path=images_dir,
        labels_path=preds_coco_labels,
        dataset_type=fiftyone.types.COCODetectionDataset,
        label_field='prediction',
    )

    if gt_coco_labels:
        gt_dataset = fiftyone.Dataset.from_dir(
            data_path=images_dir,
            labels_path=gt_coco_labels,
            dataset_type=fiftyone.types.COCODetectionDataset,
            label_field='ground_truth',
        )
        for sample in gt_dataset:
            dataset.merge_sample(sample)

    return dataset


def main(images_dir: str, preds_coco_labels: str, gt_coco_labels: Optional[str] = None) -> None:
    dataset = create_fiftyone_dataset(images_dir, preds_coco_labels, gt_coco_labels)
    session = fiftyone.launch_app(dataset)
    session.wait()


if __name__ == '__main__':
    images_dir = os.getenv('IMAGES')
    preds_path = os.getenv('PREDS')
    gt_path = os.getenv('GT')
    main(images_dir, preds_path, gt_path)  # type: ignore
