from pathlib import Path
from typing import List

import clearml
import yaml
from plotly.graph_objs import Figure
from pydantic import BaseModel

from source.train_eval.config.train_eval_cfg import TrainEvalConfig
from source.train_eval.constants import VISUALIZATIONS_TO_TRACK
from source.train_eval.evaluation.data_model import ResultsPerIou
from source.train_eval.evaluation.visualization import draw_metrics_line_plot
from source.train_eval.logger import LOGGER
from source.train_eval.pathfinding.outputs import get_output_dir


class CompleteEvalResults(BaseModel):
    results_per_iou: ResultsPerIou
    conf_thres: float
    preds_path: str
    preds_visualizations: List[str]

    def save_to_yaml(self, output_dir: Path, split: str) -> None:
        output_path = output_dir / f'metrics_eval_on_{split}.yaml'

        output_data = self.model_dump(
            exclude={'preds_visualizations': True, 'results_per_iou': {'iou_range', 'metrics_range', 'matrices_range'}},
        )
        with open(output_path, 'w') as output_file:
            yaml.dump(output_data, output_file)

        LOGGER.info(
            'Results of evaluation on %s set (metrics over a ranges of IoUs and confidence threshold value) '
            + 'are saved to %s',
            split,
            output_path,
        )

    def save_locally(self, cfg: TrainEvalConfig, split: str) -> None:
        output_dir = get_output_dir(cfg)

        self.save_to_yaml(output_dir, split)
        self.save_metrics_plots(output_dir, split)
        self.save_confusion_matrices(output_dir, split)

    def save_metrics_plots(self, output_dir: Path, split: str) -> Figure:
        figure = draw_metrics_line_plot(
            self.results_per_iou.iou_metrics_cols,
            split,
        )

        output_file = output_dir / f'{split}_metrics.png'
        figure.write_image(output_file)
        LOGGER.info('Plot of metrics on %s is saved to %s', split, output_file)

        return figure

    def save_confusion_matrices(self, output_dir: Path, split: str) -> None:
        output_dir = output_dir / f'{split}_confusion_matrices_per_iou'
        output_dir.mkdir(exist_ok=True, parents=True)

        for iou, matrix in zip(self.results_per_iou.iou_range, self.results_per_iou.matrices_range):
            matrix.savefig(str(output_dir / f'IoU_{iou}.png'))

    def track_in_clearml(self, cml_logger: clearml.Logger, split: str) -> None:
        tracking_args = (cml_logger, split)
        self._track_conf_thres(*tracking_args)
        self.results_per_iou.track_in_clearml(*tracking_args)
        self._track_metrics_plot(*tracking_args)
        self._track_preds_visualizations(*tracking_args)

    def _track_metrics_plot(self, cml_logger: clearml.Logger, split: str) -> None:
        figure = draw_metrics_line_plot(
            self.results_per_iou.iou_metrics_cols,
            split,
        )
        cml_logger.report_plotly(
            title=f'{split.capitalize()} metrics per IoU line plot',
            series='Plot',
            figure=figure,
        )

    def _track_conf_thres(self, cml_logger: clearml.Logger, split: str) -> None:
        if split == 'val':
            cml_logger.report_single_value(
                name='best_confidence_threshold',
                value=self.conf_thres,
            )

    def _track_preds_visualizations(self, cml_logger: clearml.logger, split: str) -> None:
        for img_i, img_path in enumerate(self.preds_visualizations[:VISUALIZATIONS_TO_TRACK]):
            cml_logger.report_image(
                title=f'{split}_preds_{round(self.conf_thres, 3)}_conf_thres',
                series=f'img_{img_i}',
                local_path=img_path,
            )
