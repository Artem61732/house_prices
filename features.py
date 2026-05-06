import numpy as np
import pandas as pd
from scipy.stats import skew


NONE_CATS = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
    'BsmtFinType2', 'MasVnrType', 'MSSubClass',
]

ZERO_NUMS = [
    'GarageYrBlt', 'GarageArea', 'GarageCars',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea',
]

QUAL_MAP = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

QUAL_COLS = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
    'HeatingQC', 'KitchenQual', 'FireplaceQu',
    'GarageQual', 'GarageCond', 'PoolQC',
]


def fill_na_domain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Доменная обработка пропусков для House Prices.
    NaN в этих колонках означает «нет фичи» (нет бассейна, нет гаража, и т.д.),
    а не реальный пропуск, поэтому заполняем 'None' / 0, а не средним.
    """
    df = df.copy()

    if 'MSSubClass' in df.columns:
        df['MSSubClass'] = df['MSSubClass'].astype(str)

    for c in NONE_CATS:
        if c in df.columns:
            df[c] = df[c].fillna('None')

    for c in ZERO_NUMS:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())

    return df


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет новые признаки, делает ordinal-кодирование качественных рейтингов
    и удаляет ненужные колонки.
    """
    df = data.copy()

    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)

    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['TotalBath'] = (
        df['BsmtFullBath'] + df['BsmtHalfBath']
        + df['FullBath'] + df['HalfBath']
    )

    df['TotalPorchSF'] = (
        df['OpenPorchSF'] + df['EnclosedPorch']
        + df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    )
    df['HasPool']      = (df['PoolArea'] > 0).astype(int)
    df['Has2ndFloor']  = (df['2ndFlrSF'] > 0).astype(int)
    df['HasGarage']    = (df['GarageArea'] > 0).astype(int)
    df['HasBsmt']      = (df['TotalBsmtSF'] > 0).astype(int)
    df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
    df['IsRemodeled']  = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    df['IsNew']        = (df['YrSold'] == df['YearBuilt']).astype(int)
    df['Qual_x_Area']  = df['OverallQual'] * df['GrLivArea']

    for c in QUAL_COLS:
        if c in df.columns:
            df[c] = df[c].map(QUAL_MAP).fillna(0).astype(int)

    return df


def get_skewed_columns(df: pd.DataFrame, threshold: float = 0.75) -> list[str]:
    """
    Возвращает список числовых колонок со |skew| > threshold,
    у которых все значения неотрицательные (чтобы безопасно применять log1p).
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    skewness = df[num_cols].apply(lambda x: skew(x.dropna())).abs()
    cols = skewness[skewness > threshold].index.tolist()
    return [c for c in cols if (df[c] >= 0).all()]


def log_skewed_features(df: pd.DataFrame, skewed_cols: list[str]) -> pd.DataFrame:
    """
    Применяет log1p к указанным колонкам.
    Список колонок должен быть посчитан на train (через get_skewed_columns)
    и применён согласованно к train/test.
    """
    df = df.copy()
    for c in skewed_cols:
        if c in df.columns:
            df[c] = np.log1p(df[c])
    return df


def prepare_for_catboost(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Готовит данные для нативного режима CatBoost:
    - категориальные колонки -> str + 'None' для NaN
    - числовые NaN не трогаем, CatBoost сам их обработает.
    Возвращает (X_clean, cat_features).
    """
    X = X.copy()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    for c in cat_features:
        X[c] = X[c].fillna('None').astype(str)
    return X, cat_features


def prepare_for_lightgbm(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Готовит данные для нативного режима LightGBM:
    - категориальные колонки -> pandas Categorical (LightGBM их auto-detect),
    - числовые NaN не трогаем, LightGBM сам их обработает.
    Возвращает (X_clean, cat_features).
    """
    X = X.copy()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    for c in cat_features:
        X[c] = X[c].fillna('None').astype('category')
    return X, cat_features
