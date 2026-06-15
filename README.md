# House Prices

[Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques): регрессия `SalePrice` по 79 признакам, метрика **RMSLE**.

## Установка и запуск

```bash
pip install -r requirements.txt
# положить data/train.csv и data/test.csv
python main.py
```

`python main.py` — ML: KFold CV + blend-сабмит → `submission.csv` + артефакты в `outputs/ml/`.

```bash
python main.py --quick              # быстрый ML (3 фолда, Ridge + бустинги)
python main.py --pipeline dl          # DL: CV + сабмит
python main.py --pipeline all         # ML + DL
python main.py --pipeline all --quick
python main.py --cv-only              # только CV, без сабмитов
python main.py --sub-only             # только сабмиты
```

Без `outputs/ml/best_params.json` модели обучаются с дефолтными гиперпараметрами. Для тюнинга: `python -m ml.tune`.

### Артефакты после прогона

| Файл | Описание |
|------|----------|
| `outputs/ml/results.csv` | CV-метрики всех ML-моделей |
| `outputs/ml/results.json` | то же в JSON |
| `outputs/ml/validation_report.json` | лучшая модель, leaderboard, веса blend |
| `outputs/dl/results.csv` | CV-метрики DNN-экспериментов |
| `outputs/dl/results.json` | то же в JSON |
| `submission.csv` | ML-сабмит |
| `submission_dl.csv` | DL-сабмит |

Все команды ниже — из корня репозитория.

### ML

| Задача | Команда |
|--------|---------|
| CV + сабмит | `python main.py` |
| Только CV | `python -m ml.main` |
| Только сабмит (blend) | `python -m ml.create_submission` |
| Тюнинг Optuna | `python -m ml.tune --models catboost ridge lightgbm` |

```bash
python -m ml.main --quick
python -m ml.tune --models lightgbm --n-trials 50   # по умолчанию тюнится только lightgbm
```

### DL

| Задача | Команда |
|--------|---------|
| CV + сабмит | `python main.py --pipeline dl` |
| Только CV | `python -m dl.main` |
| Только сабмит | `python -m dl.create_submission` |
| Тюнинг Optuna | `python -m dl.tune` |

```bash
python -m dl.main --quick
python -m dl.main --experiments baseline_2layer embeddings
python -m dl.create_submission --no-tuned --experiment embeddings
python -m dl.tune --n-trials 10 --n-epochs 40 --patience 8
```

Сабмиты: `submission.csv` (ML), `submission_dl.csv` (DL).

---

Два пайплайна с общим препроцессингом (`data.py`, `features.py`):
- **ML** — линейные модели, бустинги, blend
- **DL** — MLP на PyTorch

## Результаты (локальный CV)

Метрики на `log1p(SalePrice)`; RMSLE = RMSE. Числа — ориентир после `ml/tune.py` и `dl/tune.py`, воспроизводятся через `python main.py --cv-only` и `python -m dl.main`.

### ML (KFold, 5 фолдов)

| Модель | CV RMSLE |
|--------|----------|
| Linear Regression | ~0.157 |
| Ridge | ~0.133 |
| Random Forest | ~0.147 |
| XGBoost | ~0.138 |
| CatBoost | ~0.125 |
| LightGBM | ~0.124 |
| **Blend (Ridge + CatBoost + LightGBM)** | **~0.121** |

### DL (StratifiedKFold, 9 экспериментов в `dl/config.yaml`)

| Эксперимент | CV RMSLE |
|-------------|----------|
| baseline_2layer | ~0.155 |
| deep_4layer | ~0.152 |
| batchnorm | ~0.148 |
| dropout_0.2 | ~0.146 |
| embeddings | ~0.143 |
| **DNN (Optuna)** | **~0.140** |

Остальные эксперименты (`dropout_0.5`, `elu_wide`, `sgd_cosine`, `adamw_mse`) — в выводе `python -m dl.main`.

## Структура

```
house_prices/
├── main.py              # entrypoint: ML CV + сабмит
├── bootstrap.py         # sys.path для entrypoints
├── config.py            # merge config.yaml + ml/config.yaml + dl/config.yaml
├── config.yaml          # paths, random_state, cv, preprocess
├── data.py
├── features.py
├── paths.py
├── requirements.txt
├── notebooks/eda.ipynb
├── ml/
│   ├── config.yaml      # blend.weights, tune.n_trials
│   ├── main.py          # CV всех моделей
│   ├── results.py       # сохранение results.csv / validation_report.json
│   ├── create_submission.py
│   ├── tune.py
│   ├── models.py, cv.py, blend.py, train_config.py, constants.py
└── dl/
    ├── config.yaml      # dl.*, experiments, dl.tune
    ├── main.py          # CV экспериментов
    ├── results.py       # сохранение results.csv
    ├── create_submission.py
    ├── tune.py
    ├── model.py, dataset.py, train.py, train_config.py, constants.py
```

`data/`, `outputs/`, `submission*.csv`, `.venv/` — в `.gitignore`.

## Конфигурация

Три yaml сливаются в `cfg` через `OmegaConf.merge`. Параметры после тюнинга — в `outputs/ml/best_params.json` и `outputs/dl/best_params.json`.

| Что менять | Где |
|------------|-----|
| Пути к данным, `random_state`, `cv.n_splits` | `config.yaml` |
| Порог skew для `log1p` | `config.yaml` → `preprocess.skew_threshold` |
| **Веса blend** | `ml/config.yaml` → `blend.weights` (вручную, не из Optuna) |
| `n_trials` для ML-тюнинга | `ml/config.yaml` → `tune.n_trials` (30) |
| Гиперпараметры CatBoost / Ridge / LightGBM | `outputs/ml/best_params.json` |
| Дефолты и эксперименты DNN | `dl/config.yaml` |
| Лучшие гиперпараметры DNN | `outputs/dl/best_params.json` |

Веса blend для сабмита берутся из `ml/config.yaml`. Подбор весов на OOF (`ml/blend.py`) используется только при CV в `ml/main.py`.

Пример весов (`ml/config.yaml`):

```yaml
blend:
  weights:
    catboost: 0.30
    ridge:    0.45
    lightgbm: 0.25
```

## Препроцессинг

`features.preprocess()` — одинаков для ML и DL:

1. `fill_na_domain` — NaN → `"None"` / `0` по смыслу признака
2. `feature_engineering` — новые признаки, ordinal для качества
3. `log_skewed_features` — `log1p` на скошенных колонках (список с train, на test тот же)

Таргет: `y_log = log1p(SalePrice)`. В `data.py` удаляются выбросы `GrLivArea > 4000` при низкой цене.

## ML

**CV** (`ml/main.py`): Linear Regression, Ridge, Random Forest, XGBoost, CatBoost, LightGBM + blend на OOF.

**Сабмит** (`ml/create_submission.py`): blend CatBoost + Ridge + LightGBM.

**Тюнинг** (`ml/tune.py`): Optuna, KFold. По умолчанию `--models lightgbm`.

## DL

**Архитектура** (`dl/model.py`): MLP, опционально BatchNorm, Dropout, Embedding.

**Кодирование категорий** (`cat_encoding` в `dl/dataset.py`): `embedding`, `onehot`, `freq`, `target`.

**CV**: StratifiedKFold по бинам `y_log` (`dl.cv_strategy`, `dl.stratify_bins`).

**Эксперименты**: список в `dl/config.yaml` → `dl.experiments` (9 штук).

**Сабмит**: по умолчанию из `outputs/dl/best_params.json`; без него — эксперимент `embeddings` или `--experiment <name>`.

## Зависимости

См. `requirements.txt`: pandas, scikit-learn, xgboost, catboost, lightgbm, optuna, omegaconf, torch, matplotlib, seaborn.
