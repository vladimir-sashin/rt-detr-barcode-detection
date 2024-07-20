from argparse import Namespace
from pathlib import Path

from tools.train import main as rtdetr_main

from source.train_eval.config.train_eval_cfg import TrainEvalConfig
from source.train_eval.evaluation.inference import export_preds, run_inference
from source.train_eval.evaluation.metrics_calculation import (
    calculate_metrics_iou_range,
    find_optimal_conf,
)
from source.train_eval.evaluation.results_export import CompleteEvalResults
from source.train_eval.evaluation.visualization import draw_preds
from source.train_eval.logger import LOGGER
from source.train_eval.pathfinding.inputs import (
    find_dataset,
    load_coco_to_fiftyone,
)
from source.train_eval.rtdetr_configurator import (
    generate_rtdetr_configs,
    get_arch_cfg_path,
)


def evaluate_f1_conf_on_val(onnx_model: Path, cfg: TrainEvalConfig) -> CompleteEvalResults:
    split = 'val'
    LOGGER.info('Preparing for evaluation of F-1 score on val set and search for the optimal confidence threshold...')
    # 1. Find validation dataset and load it as 51 Dataset
    val_dataset = load_coco_to_fiftyone(find_dataset(cfg).valid)
    # 2. Run inference and add predicts to the dataset
    preds_dataset = run_inference(onnx_model, val_dataset, cfg.data_cfg.idx_to_class)
    # 3. Run function that optimizes F-1 by varying confidence threshold
    conf_thres = find_optimal_conf(preds_dataset)
    # 4. Calculate F-1, precision and recall for a range of IoU
    results_per_iou = calculate_metrics_iou_range(conf_thres, val_dataset, split)
    # 5. Export preds as COCO JSON labels, save predict visualizations and metrics
    eval_results = CompleteEvalResults(
        results_per_iou=results_per_iou,
        conf_thres=conf_thres,
        preds_path=str(export_preds(preds_dataset, cfg, split)),
        preds_visualizations=draw_preds(preds_dataset, cfg, conf_thres, split),
    )
    eval_results.save_locally(cfg, split)

    return eval_results


def evaluate_f1_on_test(onnx_model: Path, cfg: TrainEvalConfig, conf_thres: float) -> CompleteEvalResults:
    split = 'test'
    LOGGER.info('Preparing for evaluation of F-1 score on test set using confidence threshold = %.3f...', conf_thres)
    # 1. Find test dataset and load it as 51 Dataset
    test_dataset = load_coco_to_fiftyone(find_dataset(cfg).test)
    # 2. Run inference
    preds_dataset = run_inference(onnx_model, test_dataset, cfg.data_cfg.idx_to_class)
    # 4. Calculate F-1, precision and recall for a range of IoU
    results_per_iou = calculate_metrics_iou_range(conf_thres, preds_dataset, split)
    # 5. Export preds as COCO JSON labels, save predict visualizations and metrics
    eval_results = CompleteEvalResults(
        results_per_iou=results_per_iou,
        conf_thres=conf_thres,
        preds_path=str(export_preds(preds_dataset, cfg, split=split)),
        preds_visualizations=draw_preds(preds_dataset, cfg, conf_thres, split),
    )
    eval_results.save_locally(cfg, split)

    return eval_results


def evaluate_map_on_test(cfg: TrainEvalConfig, checkpoint_path: Path) -> None:
    test_eval_cfg_dir = generate_rtdetr_configs(cfg, eval_on_test=True)
    _evaluate_map_on_test(cfg, test_eval_cfg_dir, checkpoint_path)


def _evaluate_map_on_test(cfg: TrainEvalConfig, rtdetr_configs_dir: Path, checkpoint_path: Path) -> None:
    # Run RTDETR evaluation
    LOGGER.info('Starting mAP evaluation of the best checkpoint on test set...')
    kwargs = {
        'config': get_arch_cfg_path(cfg, rtdetr_configs_dir),
        'resume': str(checkpoint_path),
        'amp': cfg.runtime_cfg.use_amp,
        'tuning': None,
        'seed': cfg.training_cfg.seed,
        'test_only': True,
    }
    rtdetr_main(Namespace(**kwargs))
    LOGGER.info('mAP evaluation on test set is finished')
