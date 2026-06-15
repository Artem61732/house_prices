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
ML_RESULTS_CSV = ML_OUTPUTS_DIR / 'results.csv'
ML_RESULTS_JSON = ML_OUTPUTS_DIR / 'results.json'
ML_VALIDATION_REPORT = ML_OUTPUTS_DIR / 'validation_report.json'
DL_RESULTS_CSV = DL_OUTPUTS_DIR / 'results.csv'
DL_RESULTS_JSON = DL_OUTPUTS_DIR / 'results.json'


def ensure_output_dirs() -> None:
    """Создаёт outputs/ml/ и outputs/dl/ с подпапками backups/."""
    for d in (ML_OUTPUTS_DIR, DL_OUTPUTS_DIR, ML_BACKUPS_DIR, DL_BACKUPS_DIR):
        d.mkdir(parents=True, exist_ok=True)
