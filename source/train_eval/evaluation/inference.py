from pathlib import Path

import fiftyone
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision.transforms import ToTensor

from source.data.constants import PREDICT_FIELD
from source.train_eval.config.train_eval_cfg import TrainEvalConfig
from source.train_eval.constants import DEFAULT_IMG_SIZE
from source.train_eval.evaluation.data_model import Size
from source.train_eval.logger import LOGGER
from source.train_eval.pathfinding.outputs import get_output_dir


def run_inference(onnx_model_path: Path, dataset: fiftyone.Dataset, idx_to_class: dict[int, str]) -> fiftyone.Dataset:
    LOGGER.info('Running inference to get model predicts...')
    session = ort.InferenceSession(onnx_model_path)
    size = Size(*DEFAULT_IMG_SIZE)  # TODO: parametrize

    for sample in dataset:
        # Inference code is mostly borrowed from RT-DETR repo (third_party/rtdetr_pytorch/tools/export_onnx.py:84)
        model_output = _get_model_output(session, sample, size)
        detections = _postprocess_output(model_output, size, idx_to_class)
        sample[PREDICT_FIELD] = fiftyone.Detections(detections=detections)
        sample.save()

    LOGGER.info('Inference run is done, predicts are successfully obtained.')
    return dataset


def export_preds(preds_dataset: fiftyone.Dataset, cfg: TrainEvalConfig, split: str) -> Path:
    output_dir = get_output_dir(cfg)
    output_filepath = output_dir / f'{split}_preds.json'
    preds_dataset.export(
        labels_path=str(output_filepath),
        dataset_type=fiftyone.types.COCODetectionDataset,
        label_field='prediction_detections',
    )
    LOGGER.info('Predictions of best checkpoint are saved to %s', output_filepath)
    return output_filepath


def _get_model_output(session: ort.InferenceSession, sample: fiftyone.Sample, size: Size) -> np.ndarray:
    original_img = Image.open(sample.filepath).convert('RGB')
    img = original_img.resize(size)
    img_data = ToTensor()(img)[None]

    session_input = {
        'images': img_data.data.numpy(),
        'orig_target_sizes': np.array([size], dtype=np.int64),
    }
    return session.run(
        output_names=None,
        input_feed=session_input,
    )


def _normalize_xyxy_bbox(box: np.ndarray[int], size: Size) -> list[int]:
    norm_bbox = []
    for idx, coord in enumerate(box):
        norm_bbox.append(coord / size[idx % 2])

    return norm_bbox


def _xyxy_box_to_xywh(box: list[int]) -> list[int]:
    box_x, box_y = box[:2]
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]
    return [box_x, box_y, box_w, box_h]


def _postprocess_output(
    model_output: np.ndarray,
    size: Size,
    idx_to_class: dict[int, str],
) -> list[fiftyone.Detection]:
    detections = []
    for label, box, score in zip(*map(np.squeeze, model_output)):
        # TODO: introduce a class for bbox processing
        norm_box = _normalize_xyxy_bbox(box, size)
        xywh_norm_box = _xyxy_box_to_xywh(norm_box)
        detections.append(
            fiftyone.Detection(
                label=idx_to_class[label],
                bounding_box=xywh_norm_box,
                confidence=score,
            ),
        )

    return detections
