from functools import partial
from typing import Optional

from pycocotools.cocoeval import COCOeval
from torch.utils.tensorboard import SummaryWriter

from misc import MetricLogger

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
