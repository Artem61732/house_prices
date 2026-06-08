# House Prices — ML и DL пайплайны

Проект для соревнования [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques): предсказание цены дома (`SalePrice`) по табличным признакам.

Два независимых пайплайна:
- **ML** — CatBoost, Ridge, LightGBM и blend (основной скор на LB).
- **DL** — DNN (MLP) на PyTorch (эксперименты и сравнение с ML).

Общие модули (загрузка данных, feature engineering, конфиг) лежат в корне и переиспользуются обоими пайплайнами.

---

## Структура проекта

```
house_prices/
├── bootstrap.py           # sys.path — импортировать первым в entrypoints
├── paths.py               # outputs/ml/, outputs/dl/ — артефакты обучения
├── config.py              # слияние config.yaml + ml/config.yaml + dl/config.yaml
├── config.yaml            # общее: paths, random_state, cv, preprocess
├── data.py                # load_data(), load_preprocessed_*()
├── features.py            # доменный FE, preprocess(), prepare_for_*
├── requirements.txt
├── data/                  # train.csv, test.csv (не в git)
├── outputs/               # артефакты (не в git)
│   ├── ml/
│   │   ├── best_params.json
│   │   └── backups/
│   └── dl/
│       ├── best_params.json
│       └── backups/
├── notebooks/
│   └── eda.ipynb
├── ml/                    # ML-пайплайн
│   ├── config.yaml        # blend.weights (ручные), tune (Optuna ML)
│   ├── constants.py       # CATBOOST_FIXED, LIGHTGBM_FIXED
│   ├── models.py          # get_preprocessor, фабрики моделей
│   ├── cv.py              # CV-хелперы
│   ├── blend.py           # find_blend_weights
│   ├── train_config.py    # load_tuned_params, load_best_params
│   ├── main.py            # оценка моделей (KFold CV) + blend
│   ├── tune.py            # Optuna-тюнинг CatBoost / Ridge / LightGBM
│   └── create_submission.py
└── dl/                    # DL-пайплайн
    ├── config.yaml        # dl.*, experiments, dl.tune
    ├── train_config.py    # TrainConfig, build_train_config_*, load_tuned_params
    ├── constants.py       # CAT_ENCODINGS, search space для Optuna
    ├── model.py           # HousePriceMLP
    ├── dataset.py         # FeatureEncoder, DataLoader
    ├── train.py           # обучение, CV, predict
    ├── main.py            # evaluate_dnn_experiments()
    ├── tune.py            # Optuna-тюнинг DNN
    └── create_submission.py
```

---

## Установка

Из корня проекта:

```bash
pip install -r requirements.txt
```

Данные Kaggle положить в `data/`:
- `data/train.csv`
- `data/test.csv`

---

## Быстрый старт

Все команды запускаются **из корня** `house_prices/`:

| Задача | Команда |
|--------|---------|
| Оценка ML-моделей (CV) | `python -m ml.main` |
| Сабмит ML (blend) | `python -m ml.create_submission` |
| Тюнинг ML (Optuna) | `python -m ml.tune --models lightgbm` |
| Оценка DL-экспериментов | `python -m dl.main` |
| Сабмит DL (из best_params) | `python -m dl.create_submission` |
| Тюнинг DL (Optuna) | `python -m dl.tune` |

Результаты:
- ML → `submission.csv`
- DL → `submission_dl.csv`

---

## Конфигурация — источник правды

Три yaml-файла собираются в **`config.py`** через `OmegaConf.merge`.  
Отдельно от yaml есть Python-модули **`ml/train_config.py`** и **`dl/train_config.py`** — они читают/собирают параметры обучения и `best_params.json`.

| Файл | Что здесь | Кто читает |
|------|-----------|------------|
| **`config.yaml`** | `paths`, `random_state`, `cv`, `preprocess` | ML + DL |
| **`ml/config.yaml`** | `blend.weights`, `tune` (Optuna ML) | ML |
| **`dl/config.yaml`** | `dl.*`, `dl.experiments`, `dl.tune` | DL |

### Что можно менять и откуда берётся

| Параметр | Файл | Ручной / автоматический |
|----------|------|-------------------------|
| Пути к данным, сабмитам | `config.yaml` → `paths` | ручной |
| `random_state`, `cv.n_splits` | `config.yaml` | ручной |
| `preprocess.skew_threshold` | `config.yaml` | ручной |
| **Веса ML-blend** | `ml/config.yaml` → `blend.weights` | **ручной** (не из тюнинга!) |
| Гиперпараметры CatBoost/Ridge/LGBM | `outputs/ml/best_params.json` | Optuna (`ml/tune.py`) |
| Дефолты Optuna ML | `ml/config.yaml` → `tune.n_trials` | ручной |
| Дефолты DNN | `dl/config.yaml` → `dl.*` | ручной |
| DL-эксперименты | `dl/config.yaml` → `dl.experiments` | ручной (учебные сценарии) |
| Лучшие гиперпараметры DNN | `outputs/dl/best_params.json` | Optuna (`dl/tune.py`) |

**Важно:** `blend.weights` задаются вручную в `ml/config.yaml`.  
`outputs/ml/best_params.json` содержит только гиперпараметры моделей, не веса blend.  
Подбор весов на OOF есть в `ml/main.py` (`find_blend_weights`) — для анализа, не для сабмита.

### Примеры

Изменить веса blend перед ML-сабмитом (`ml/config.yaml`):

```yaml
blend:
  weights:
    catboost: 0.30
    ridge:    0.45
    lightgbm: 0.25
```

Добавить DL-эксперимент (`dl/config.yaml` → `dl.experiments`):

```yaml
- name: "my_experiment"
  hidden_layers: [256, 128]
  cat_encoding: "onehot"
  dropout: 0.2
```

---

## Общий препроцессинг

Цепочка в `features.preprocess()` (одинакова для ML и DL):

1. **`fill_na_domain`** — доменные NaN (`None` / `0`, не среднее).
2. **`feature_engineering`** — новые признаки, ordinal-кодирование качества.
3. **`log_skewed_features`** — `log1p` на скошенных числовых колонках.

На **train** список скошенных колонок считается автоматически.  
На **test** передаётся тот же список с train (чтобы не было leakage).

Таргет для обучения: `y_log = log1p(SalePrice)`.  
Метрика соревнования RMSLE на лог-таргете совпадает с RMSE.

---

## ML-пайплайн

### Оценка (`ml/main.py`)

- KFold CV на `y_log`.
- Модели: Linear Regression, Ridge, Random Forest, XGBoost, CatBoost (native), LightGBM (native).
- Blend Ridge + CatBoost + LightGBM: равные веса и перебор лучших весов на OOF.
- Параметры Ridge / CatBoost / LightGBM подхватываются из `outputs/ml/best_params.json`, если файл есть.

### Тюнинг (`ml/tune.py`)

```bash
python -m ml.tune --models catboost ridge lightgbm
python -m ml.tune --models lightgbm --n-trials 50
```

Лучшие параметры сохраняются в `outputs/ml/best_params.json` (с бэкапом в `outputs/ml/backups/`).

### Сабмит (`ml/create_submission.py`)

Blend трёх моделей с весами из `config.yaml` и параметрами из `outputs/ml/best_params.json`:

```bash
python -m ml.create_submission
```

---

## DL-пайплайн

### Архитектура

- **MLP** (`dl/model.py`) с опциональными BatchNorm, Dropout, Embedding.
- **Кодирование категорий** (`cat_encoding` в `dl/dataset.py`):
  - `embedding` — отдельные Embedding-слои
  - `onehot` — OneHotEncoder, конкатенация с числовыми
  - `freq` — частотное кодирование
  - `target` — target encoding (fit только на train-фолде в CV)

### CV

StratifiedKFold по квантильным бинам `y_log` (настраивается в `dl/config.yaml`: `dl.cv_strategy`, `dl.stratify_bins`).

### Эксперименты (`dl/main.py`)

Список конфигов в `dl/config.yaml` → секция `dl.experiments`.

```bash
# все эксперименты
python -m dl.main

# выбранные
python -m dl.main --experiments baseline_2layer embeddings
```

### Тюнинг (`dl/tune.py`)

```bash
# refined search space (по умолчанию, 50 trials из config)
python -m dl.tune

# быстрый прогон
python -m dl.tune --n-trials 10 --n-epochs 40 --patience 8

# широкий поиск (exploratory)
python -m dl.tune --search-space wide --n-trials 30
```

Результат → `outputs/dl/best_params.json` (ключ `dnn`, метрика `dnn_cv_rmsle`).

### Сабмит (`dl/create_submission.py`)

```bash
# из outputs/dl/best_params.json (по умолчанию)
python -m dl.create_submission

# из эксперимента dl/config.yaml
python -m dl.create_submission --no-tuned --experiment embeddings
```

---

## Артефакты и git

| Файл | Описание |
|------|----------|
| `submission.csv` | ML-сабмит |
| `submission_dl.csv` | DL-сабмит |
| `outputs/ml/best_params.json` | тюненные параметры ML |
| `outputs/dl/best_params.json` | тюненные параметры DNN |
| `outputs/*/backups/` | timestamped-бэкапы после тюнинга |

`data/`, `outputs/`, `submission*.csv`, `__pycache__/` — в `.gitignore`.

---

## Публичный API (для code review)

| Модуль | Функции | Назначение |
|--------|---------|------------|
| `data.py` | `load_data()`, `load_preprocessed_train_target()` | загрузка и подготовка train |
| `data.py` | `load_preprocessed_dl_train_target()` | то же + списки колонок для DNN |
| `ml/__init__.py` | `get_preprocessor`, `find_blend_weights`, `load_tuned_params` | публичный API ML |
| `ml/train_config.py` | `load_best_params()` | чтение best_params для сабмита |
| `dl/__init__.py` | `TrainConfig`, `build_train_config_*`, `load_tuned_params` | публичный API DL |
| `dl/constants.py` | `CAT_ENCODINGS`, `REFINED_*`, `WIDE_*` | search space Optuna |
| `dl/main.py` | `evaluate_dnn_experiments()` | KFold CV для экспериментов |

Функции с префиксом `_` — внутренние, не импортировать из других пакетов.

## Заметки для code review

1. **Границы модулей:** общая логика — в корне (`data.py`, `features.py`, `bootstrap.py`, `paths.py`); пайплайн-специфичное — в `ml/` или `dl/`.
2. **Точки входа:** `python -m ml.*` / `python -m dl.*` из корня; первый импорт — `bootstrap`.
3. **Leakage:** `skewed_cols` и target encoding считаются только на train (или train-фолде).
4. **ML vs DL:** ML — основной продакшен-пайплайн; DL — отдельная ветка с собственным `best_params.json` и сабмитом.
5. **Конфиг:** три yaml → `config.py`; TrainConfig и best_params — через `ml/train_config.py` / `dl/train_config.py`.

---

## Зависимости

- **ML:** pandas, scikit-learn, xgboost, catboost, lightgbm, optuna
- **DL:** torch
- **Конфиг:** omegaconf

Полный список — в `requirements.txt`.
