import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, root_mean_squared_log_error, mean_absolute_error
from sklearn.impute import SimpleImputer

test_df = pd.read_csv('data/test.csv')
train_df = pd.read_csv('data/train.csv')
simpleImputer = SimpleImputer(missing_values = np.nan)

X = train_df.drop('SalePrice', axis=1)
X = pd.get_dummies(X)
X = simpleImputer.fit_transform(X)

y = train_df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred = np.maximum(0, y_pred)

mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
rmsle = root_mean_squared_log_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R2 Score: {r2}')
print(f'Root Mean Squared Log Error: {rmsle}')