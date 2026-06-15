"""Поиск весов blend на OOF-предсказаниях."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import root_mean_squared_error


def find_blend_weights(oof_preds: dict, y_log, step: float = 0.1) -> dict:
    """
    Перебор весов на сетке с шагом step (сумма весов = 1).
    Используется в ml/main.py для анализа; веса сабмита — из ml/config.yaml.
    """
    names = list(oof_preds)
    arrs = [np.asarray(oof_preds[n]) for n in names]
    y_arr = np.asarray(y_log)
    n = len(names)
    grid = np.arange(0.0, 1.0 + 1e-9, step)

    candidates = []
    if n == 3:
        for w0 in grid:
            for w1 in grid:
                w2 = round(1.0 - w0 - w1, 2)
                if w2 < -1e-9:
                    continue
                weights = (round(float(w0), 2), round(float(w1), 2), max(w2, 0.0))
                pred = sum(w * a for w, a in zip(weights, arrs))
                score = float(root_mean_squared_error(y_arr, pred))
                candidates.append((weights, score))
    else:
        raise ValueError(f"find_blend_weights supports 3 models, got {n}")

    candidates.sort(key=lambda t: t[1])
    best_weights, best_score = candidates[0]
    equal_weights = (1.0 / n,) * n
    equal_pred = sum(w * a for w, a in zip(equal_weights, arrs))
    equal_score = float(root_mean_squared_error(y_arr, equal_pred))

    return {
        'names': names,
        'best_weights': best_weights,
        'best_score': best_score,
        'equal_weights': equal_weights,
        'equal_score': equal_score,
        'top5': candidates[:5],
    }
