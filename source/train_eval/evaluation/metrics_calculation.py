import warnings

import fiftyone
import scipy
from fiftyone import DetectionResults

from source.data.constants import GT_FIELD, PREDICT_FIELD
from source.train_eval.constants import (
    CONF_SEARCH_IOU,
    CONF_SEARCH_LOWER_BOUND,
    CONF_SEARCH_UPPER_BOUND,
    CONF_SEARCH_XTOL,
    IOU_RANGE,
)
from source.train_eval.evaluation.data_model import (
    Metrics,
    ResultsPerIou,
    SingleEvalResult,
)
from source.train_eval.evaluation.metrics_utils import filter_preds_by_conf
from source.train_eval.logger import LOGGER


def evaluate_metrics(
    conf: float,
    preds_dataset: fiftyone.Dataset,
    iou_thres: float = 0.5,
    pred_field: str = PREDICT_FIELD,
    gt_field: str = GT_FIELD,
) -> SingleEvalResult:
    eval_results = _evaluate_metrics(conf, preds_dataset, iou_thres, pred_field, gt_field)

    metrics_dict = eval_results.metrics()
    metrics_dict['f1_score'] = metrics_dict.pop('fscore')
    metrics_dict = {name: round(metric, 3) for name, metric in metrics_dict.items()}

    metrics = Metrics(**metrics_dict)
    with warnings.catch_warnings():
        # Fiftyone spams warnings that interactive plots are available only in Jupyter, which is irrelevant
        warnings.filterwarnings('ignore')
        confusion_matrix = eval_results.plot_confusion_matrix(include_missing=True, backend='matplotlib')

    return SingleEvalResult(iou=iou_thres, metrics=metrics, confusion_matrix=confusion_matrix)


def calculate_metrics_iou_range(conf: float, preds_dataset: fiftyone.Dataset, split: str) -> ResultsPerIou:
    LOGGER.info('Calculating %s F-1 scores over a ranges of IoUs with confidence threshold of %.3f...', split, conf)
    eval_results = []
    LOGGER.info('%s Metrics over a ranges of IoUs with confidence threshold of %.3f:', split, conf)
    for iou in IOU_RANGE:
        curr_eval_result = evaluate_metrics(conf, preds_dataset, iou)
        eval_results.append(curr_eval_result)
        LOGGER.info('IoU:%.2f: %s', iou, curr_eval_result.metrics)

    return ResultsPerIou.from_single_results(eval_results)


def _evaluate_metrics(
    conf: float,
    preds_dataset: fiftyone.Dataset,
    iou_thres: float = 0.5,
    pred_field: str = PREDICT_FIELD,
    gt_field: str = GT_FIELD,
) -> DetectionResults:
    conf_view = filter_preds_by_conf(preds_dataset, conf, pred_field)
    return conf_view.evaluate_detections(
        pred_field,
        gt_field=gt_field,
        eval_key='eval',
        iou=iou_thres,
        classwise=False,
    )


def find_optimal_conf(preds_dataset: fiftyone.Dataset) -> float:
    LOGGER.info('Searching for the optimal confidence threshold to maximize F-1 on val set...')
    # Confidence threshold optimization code is partially borrowed from
    # https://github.com/danielgural/optimal_confidence_threshold/
    best_conf, f1_val, _, _ = scipy.optimize.fminbound(
        func=calculate_f1,
        x1=CONF_SEARCH_LOWER_BOUND,
        x2=CONF_SEARCH_UPPER_BOUND,
        args=(preds_dataset, CONF_SEARCH_IOU),
        xtol=CONF_SEARCH_XTOL,
        full_output=True,
    )
    f1_val = -1.0 * f1_val
    best_conf = best_conf.item()
    LOGGER.info('Best confidence threshold found: %.3f', best_conf)
    LOGGER.info('Validation F-1 score @0.50: %.3f', f1_val)

    return round(best_conf, 3)


def calculate_f1(
    conf: float,
    preds_dataset: fiftyone.Dataset,
    iou_thres: float = 0.5,
    pred_field: str = PREDICT_FIELD,
    gt_field: str = GT_FIELD,
) -> float:
    eval_results = _evaluate_metrics(conf, preds_dataset, iou_thres, pred_field, gt_field)

    metrics = eval_results.metrics()
    return -metrics['fscore']
