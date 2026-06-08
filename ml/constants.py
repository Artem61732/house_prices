"""Константы и search space для ML-пайплайна (Optuna-тюнинг)."""

TUNABLE_MODELS = ('catboost', 'ridge', 'lightgbm')

CATBOOST_FIXED = dict(
    iterations=1500,
    random_state=42,
    verbose=False,
    thread_count=-1,
    od_type='Iter',
    od_wait=30,
    allow_writing_files=False,
)

LIGHTGBM_FIXED = dict(
    n_estimators=2000,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)
