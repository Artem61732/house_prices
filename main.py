"""
Единая точка входа: ML / DL / оба пайплайна.

Запуск:
    python main.py                    # ML: CV + сабмит
    python main.py --pipeline all     # ML + DL
    python main.py --quick            # быстрый ML (3 фолда, меньше моделей)
    python main.py --pipeline dl --quick
    python main.py --cv-only          # без сабмитов
    python main.py --sub-only         # только сабмиты
"""

from __future__ import annotations

import argparse
import sys

import bootstrap  # noqa: F401

from config import cfg
from paths import ensure_output_dirs


def run_ml(*, cv: bool, submission: bool, quick: bool) -> None:
    if cv:
        print("=" * 60)
        print("ML: оценка моделей (KFold CV)")
        print("=" * 60)
        from ml.main import run_evaluation
        run_evaluation(
            random_state=int(cfg.random_state),
            quick=quick,
            save=True,
        )

    if submission:
        print()
        print("=" * 60)
        print("ML: создание сабмита (blend)")
        print("=" * 60)
        from ml.create_submission import create_submission
        create_submission()


def run_dl(*, cv: bool, submission: bool, quick: bool) -> None:
    if cv:
        print("=" * 60)
        print("DL: оценка экспериментов (KFold CV)")
        print("=" * 60)
        from dl.main import evaluate_dnn_experiments
        evaluate_dnn_experiments(quick=quick, save=True)

    if submission:
        print()
        print("=" * 60)
        print("DL: создание сабмита")
        print("=" * 60)
        from dl.create_submission import create_submission
        create_submission()


def main(
    pipeline: str = 'ml',
    *,
    cv: bool = True,
    submission: bool = True,
    quick: bool = False,
) -> int:
    ensure_output_dirs()

    if pipeline in ('ml', 'all'):
        run_ml(cv=cv, submission=submission, quick=quick)

    if pipeline in ('dl', 'all'):
        if pipeline == 'all' and cv:
            print()
        run_dl(cv=cv, submission=submission, quick=quick)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="House Prices — ML / DL pipeline")
    parser.add_argument(
        '--pipeline',
        choices=('ml', 'dl', 'all'),
        default='ml',
        help="ml (default): CV + blend-сабмит; dl: DNN; all: оба",
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help="Быстрый прогон: меньше фолдов/моделей (ML) или один эксперимент (DL)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--cv-only',
        action='store_true',
        help="Только кросс-валидация (без сабмитов)",
    )
    group.add_argument(
        '--sub-only',
        action='store_true',
        help="Только сабмиты (без CV)",
    )
    args = parser.parse_args()

    run_cv = not args.sub_only
    run_sub = not args.cv_only

    sys.exit(main(
        pipeline=args.pipeline,
        cv=run_cv,
        submission=run_sub,
        quick=args.quick,
    ))
