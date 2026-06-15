"""
Единая точка входа: CV-оценка всех ML-моделей + создание сабмита.

Запуск:
    python main.py           # полный пайплайн (оценка + сабмит)
    python main.py --cv-only # только кросс-валидация, без сабмита
    python main.py --sub-only# только сабмит, без CV

Данные: data/train.csv, data/test.csv (положить вручную из Kaggle).
Результат: submission.csv (ML blend), вывод RMSLE в консоль.
"""

from __future__ import annotations

import argparse

import bootstrap  # noqa: F401

from config import cfg


def main(cv: bool = True, submission: bool = True) -> None:
    """Запускает ML-пайплайн: KFold CV и/или создание blend-сабмита."""
    if cv:
        print("=" * 60)
        print("ШАГ 1 / 2: Оценка моделей (KFold CV)")
        print("=" * 60)
        from ml.main import run_evaluation
        run_evaluation(
            n_splits=int(cfg.cv.n_splits),
            random_state=int(cfg.random_state),
        )

    if submission:
        print()
        print("=" * 60)
        print("ШАГ 2 / 2: Создание сабмита (blend)")
        print("=" * 60)
        from ml.create_submission import create_submission
        create_submission()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="House Prices — полный ML-пайплайн"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--cv-only",
        action="store_true",
        help="Только кросс-валидация (без создания сабмита)",
    )
    group.add_argument(
        "--sub-only",
        action="store_true",
        help="Только создание сабмита (без CV)",
    )
    args = parser.parse_args()

    run_cv = not args.sub_only
    run_sub = not args.cv_only

    main(cv=run_cv, submission=run_sub)
