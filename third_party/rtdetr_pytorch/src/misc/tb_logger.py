from functools import partial
from typing import Optional

import torch
import torchvision
from pycocotools.cocoeval import COCOeval
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import box_convert
from torchvision.transforms.v2 import functional
from torchvision.utils import draw_bounding_boxes

from src.misc import MetricLogger

COCO_STAT_NAMES = (
    "AP/IoU:0.50-0.95_all_100",
    "AP/IoU:0.50_all_100",
    "AP/IoU:0.75_all_100",
    "AP/IoU:0.50-0.95_small_100",
    "AP/IoU:0.50-0.95_medium_100",
    "AP/IoU:0.50-0.95_large_100",
    "AR/IoU:0.50-0.95_all_1",
    "AR/IoU:0.50-0.95_all_10",
    "AR/IoU:0.50-0.95_all_100",
    "AR/IoU:0.50-0.95_small_100",
    "AR/IoU:0.50-0.95_medium_100",
    "AR/IoU:0.50-0.95_large_100",
)


def _reformat_loss_name(input_string):
    parts = input_string.split('_')
    if len(parts) == 2:
        return '/'.join(parts)
    elif len(parts) > 2:
        return '_'.join(parts[:2]) + '/' + '_'.join(parts[2:])
    else:
        return input_string


class TBWriter(SummaryWriter):

    def __init__(self, log_dir: str, train_bbox_fmt: str):
        super().__init__(log_dir)

        self.train_bbox_fmt = train_bbox_fmt

    def log_metrics(self, metric_logger: MetricLogger, epoch: Optional[int], mode='train'):
        for name, value in metric_logger.meters.items():
            if name.startswith('loss_'):
                name = _reformat_loss_name(name)
            elif name.startswith('lr_'):
                name = name.replace('_', '/')
            self.add_scalar(f'{mode}_{name}', value.global_avg, epoch)

    def log_coco_eval(self, coco_evaluator: COCOeval, epoch: Optional[int], mode='val'):
        stats = coco_evaluator.stats
        log_stat = partial(self.add_scalar, global_step=epoch)
        for i, stat_name in enumerate(COCO_STAT_NAMES):
            log_stat(f'{mode}_{stat_name}', stats[i])

    def log_batch(self, images, targets, epoch, mode):
        if epoch is None:
            epoch = 0
        if epoch % 10 != 0:  # TODO: parametrize
            return
        annotated_images = []
        for image, target in zip(images, targets):
            image = functional.convert_dtype(image, torch.uint8)
            bbox = target['boxes']
            if mode == 'train':
                bbox = box_convert(bbox, self.train_bbox_fmt, 'xyxy')
                height, width = image.shape[1], image.shape[2]
                bbox = bbox * torch.Tensor([width, height, width, height])
            ann_img = draw_bounding_boxes(image, bbox, colors="red", width=3)
            annotated_images.append(ann_img)
        grid = torchvision.utils.make_grid(annotated_images)
        self.add_image(f'{mode}_batch', grid, global_step=epoch)
