"""
Best Prepare Data for KNN
"""
# Лучшая подготовка данных для KNN
# Rescale Data: - работает намного лучше, если все данные имеют одинаковый масштаб.
# Address Missing Data: - отсутствие данных означает, что расстояние между образцами не может быть рассчитано.
# Lower Dimensionality: - подходит для данных более низкой размерности(не много парамтеров).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.style.use('fivethirtyeight')

# Get the Data
data = pd.read_csv("data.csv")
data.columns = data.columns.str.lower()

# Exploratory Data Analysis
# pd.set_option('display.float_format', '{:.4}'.format)
# data.describe()
# data.info()
# data.isnull().sum()

# Visualization
# plt.figure(figsize=(10, 8))
# sns.scatterplot('room_board', 'grad_rate', data=data, hue='private')

# plt.figure(figsize=(10, 8))
# sns.scatterplot('outstate', 'f_undergrad', data=data, hue='private')

# plt.figure(figsize=(12, 8))
# data.loc[data.private == 'Yes', 'outstate'].hist(label="Private College", bins=30)
# data.loc[data.private == 'No', 'outstate'].hist(label="Non Private College", bins=30)
# plt.xlabel('Outstate')
# plt.legend()

# plt.figure(figsize=(12, 8))
# data.loc[data.private == 'Yes', 'grad_rate'].hist(label="Private College", bins=30)
# data.loc[data.private == 'No', 'grad_rate'].hist(label="Non Private College", bins=30)
# plt.xlabel('Graduation Rate')
# plt.legend()
# plt.show()

# Пример: Частная школа с числом выпускников выше 100%.
# school = data.loc[data.grad_rate > 100]
# Чтобы не было ошибки:
# data.loc[data.grad_rate > 100, 'grad_rate'] = 100
# plt.figure(figsize=(12, 8))
# data.loc[data.private == 'Yes', 'grad_rate'].hist(label="Private College", bins=30)
# data.loc[data.private == 'No', 'grad_rate'].hist(label="Non Private College", bins=30)
# plt.xlabel('Graduation Rate')
# plt.legend()
# plt.show()

"""
Train test split
"""
data.private.value_counts()

"""
Standardize the Variables
"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X = data.drop(['private'], axis=1)
y = data.private
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

"""
Predictions and Evaluations
"""
# Оценка модели
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
def evaluate(model, X_train, X_test, y_train, y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    print("TRAINIG RESULTS:")
    clf_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_train, y_train_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")
    print("TESTING RESULTS:")
    clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")

from sklearn.neighbors import KNeighborsClassifier
# knn_clf = KNeighborsClassifier(n_neighbors=2)
# knn_clf.fit(X_train, y_train)
# evaluate(knn_clf, X_train, X_test, y_train, y_test)

# Цикл для обученич различных моделей KNN с разными значениями k и отслеживание error_rate для каждой модели.
scores = []
for n in range(2, 40):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
# График по циклу
# plt.figure(figsize=(8, 6))
# plt.plot(range(2, 40), scores)
# plt.ylabel("Accuracy")
# plt.xlabel("K nearest neighbors")

# Повторная тренировка с новым значением K
# Переобучить модель, выбрав лучшее значение K(нужные гиперпараметры)
knn_clf = KNeighborsClassifier(n_neighbors=7)
knn_clf.fit(X_train, y_train)
evaluate(knn_clf, X_train, X_test, y_train, y_test)