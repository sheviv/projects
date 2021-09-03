import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas

# Check data
USAhousing = pd.read_csv('USA_Housing.csv')

# Разделение данныч на массив X(с функцией для обучения), У(с целевой переменной - столбец Price).
# Убрать столбец Address,т.к. содержит только текстовую информацию.
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age',
                'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
                'Area Population']]
y = USAhousing['Price']

# Train Test Split
# Разделение данныч на обучающий набор и набор для тестирования.
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()
def print_evaluate(true, predicted):
    """
    Loss functions for X,y data
    :param true:
    :param predicted:
    :return:
    """
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('std_scalar', StandardScaler())])
X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

"""
Polynomial Regression
"""
# Паттерн МО - использование линейных моделей, обученных нелинейным функциям данных.
# Пример: расширяется путем построения полиномиальных функций из коэффициентов.
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_train_2_d = poly_reg.fit_transform(X_train)
X_test_2_d = poly_reg.transform(X_test)
lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train_2_d, y_train)
test_pred = lin_reg.predict(X_test_2_d)
train_pred = lin_reg.predict(X_train_2_d)
print('Test set evaluation:')
print_evaluate(y_test, test_pred)
print('Train set evaluation:')
print_evaluate(y_train, train_pred)
