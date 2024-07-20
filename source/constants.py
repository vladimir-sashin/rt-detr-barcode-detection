from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIGS = PROJECT_ROOT / 'configs'
TRAIN_CFG = CONFIGS / 'train_eval.yaml'
DATA_CFG = CONFIGS / 'data.yaml'

DATA_DIR = PROJECT_ROOT / 'datasets'
PRETRAINED_DIR = PROJECT_ROOT / 'pretrained'
OUTPUT_DIR = PROJECT_ROOT / 'experiments'
