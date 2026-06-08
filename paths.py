"""Пути к артефактам обучения (outputs/)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / 'outputs'
ML_OUTPUTS_DIR = OUTPUTS_DIR / 'ml'
DL_OUTPUTS_DIR = OUTPUTS_DIR / 'dl'
ML_BEST_PARAMS_PATH = ML_OUTPUTS_DIR / 'best_params.json'
DL_BEST_PARAMS_PATH = DL_OUTPUTS_DIR / 'best_params.json'
ML_BACKUPS_DIR = ML_OUTPUTS_DIR / 'backups'
DL_BACKUPS_DIR = DL_OUTPUTS_DIR / 'backups'


def ensure_output_dirs() -> None:
    for d in (ML_OUTPUTS_DIR, DL_OUTPUTS_DIR, ML_BACKUPS_DIR, DL_BACKUPS_DIR):
        d.mkdir(parents=True, exist_ok=True)
