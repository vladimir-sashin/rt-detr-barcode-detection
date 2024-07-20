from functools import cache
from typing import List

import fiftyone
import pandas as pd
from fiftyone.utils.annotations import DrawConfig
from plotly import express as px
from plotly.graph_objs import Figure

from source.train_eval.config.train_eval_cfg import TrainEvalConfig
from source.train_eval.evaluation.data_model import IousMetricsColumns
from source.train_eval.evaluation.metrics_utils import filter_preds_by_conf
from source.train_eval.logger import LOGGER
from source.train_eval.pathfinding.outputs import get_output_dir


@cache
def draw_metrics_line_plot(iou_metrics_cols: IousMetricsColumns, split: str) -> Figure:
    ious, metrics_names, metrics_values = iou_metrics_cols

    df_dict = {
        'IoU': ious,
        'metric': metrics_names,
        'value': metrics_values,
    }
    df = pd.DataFrame(df_dict)
    title = f'Metrics on {split} set'
    return px.line(df, x='IoU', y='value', color='metric', title=title)


def draw_preds(preds_dataset: fiftyone.Dataset, cfg: TrainEvalConfig, conf_thres: float, split: str) -> List[str]:
    LOGGER.info(
        'Drawing visualizations of predicts and GT on %s set with confidence threshold of %.3f...',
        split,
        conf_thres,
    )
    output_path = get_output_dir(cfg) / f'visualizations_{split}'
    output_path.mkdir(exist_ok=True, parents=True)
    conf_view = filter_preds_by_conf(preds_dataset, conf_thres)
    draw_cfg = DrawConfig(
        {
            'show_object_attrs': False,
            'show_object_confidences': True,
        },
    )
    visualization_paths = conf_view.draw_labels(output_path, label_fields=None, config=draw_cfg)
    LOGGER.info('Visualizations are saved to %s', output_path)
    return visualization_paths
