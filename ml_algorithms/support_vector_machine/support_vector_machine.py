import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Check data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
col_names = list(cancer.feature_names)
col_names.append('target')
df = pd.DataFrame(np.c_[cancer.data, cancer.target], columns=col_names)

# df.head()
# print(cancer.target_names)
# df.describe()
# df.info()

"""
VISUALIZING THE DATA
"""
# print(df.columns)
# sns.pairplot(df, hue='target', vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
#                                      'mean smoothness', 'mean compactness', 'mean concavity',
#                                      'mean concave points', 'mean symmetry', 'mean fractal dimension'])

# sns.countplot(df['target'], label="Count")

# plt.figure(figsize=(10, 8))
# sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df)

# Корреляция между переменными
# Сильная корреляция между средним радиусом и средним периметром, средней площадью и средним примером
# plt.figure(figsize=(20,10))
# sns.heatmap(df.corr(), annot=True)

# plt.show()

"""
MODEL TRAINING
"""
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

X = df.drop('target', axis=1)
y = df.target
print(f"'X' shape: {X.shape}")
print(f"'y' shape: {y.shape}")
pipeline = Pipeline([
    ('min_max_scaler', MinMaxScaler()),
    ('std_scaler', StandardScaler())
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
    elif train == False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# Linear Kernel SVM
from sklearn.svm import LinearSVC
# model = LinearSVC(loss='hinge', dual=True)
# model.fit(X_train, y_train)
# print_score(model, X_train, y_train, X_test, y_test, train=True)
# print_score(model, X_train, y_train, X_test, y_test, train=False)

# Polynomial Kernel SVM
from sklearn.svm import SVC
# Гиперпараметр coef0 контролирует, насколько модель подвержена влиянию многочленов высокой степени.
# model = SVC(kernel='poly', degree=2, gamma='weather_forecasting', coef0=1, C=5)
# model.fit(X_train, y_train)
# print_score(model, X_train, y_train, X_test, y_test, train=True)
# print_score(model, X_train, y_train, X_test, y_test, train=False)

# Radial Kernel SVM
# model = SVC(kernel='rbf', gamma=0.5, C=0.1)
# model.fit(X_train, y_train)
# print_score(model, X_train, y_train, X_test, y_test, train=True)
# print_score(model, X_train, y_train, X_test, y_test, train=False)

"""
Data Preparation for SVM
"""
# X_train = pipeline.fit_transform(X_train)
# X_test = pipeline.transform(X_test)
# print("=======================Linear Kernel SVM==========================")
# model = SVC(kernel='linear')
# model.fit(X_train, y_train)
# print_score(model, X_train, y_train, X_test, y_test, train=True)
# print_score(model, X_train, y_train, X_test, y_test, train=False)
# print("=======================Polynomial Kernel SVM==========================")
# from sklearn.svm import SVC
# model = SVC(kernel='poly', degree=2, gamma='weather_forecasting')
# model.fit(X_train, y_train)
# print_score(model, X_train, y_train, X_test, y_test, train=True)
# print_score(model, X_train, y_train, X_test, y_test, train=False)
# print("=======================Radial Kernel SVM==========================")
# from sklearn.svm import SVC
# model = SVC(kernel='rbf', gamma=1)
# model.fit(X_train, y_train)
# print_score(model, X_train, y_train, X_test, y_test, train=True)
# print_score(model, X_train, y_train, X_test, y_test, train=False)

"""
Support Vector Machine Hyper parameter tuning
"""
from sklearn.model_selection import GridSearchCV
# param_grid = {'C': [0.01, 0.1, 0.5, 1, 10, 100],
#               'gamma': [1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001],
#               'kernel': ['rbf', 'poly', 'linear']}
# grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5, iid=True)
# grid.fit(X_train, y_train)
# best_params = grid.best_params_
# print(f"Best params: {best_params}")
# svm_clf = SVC(**best_params)
# svm_clf.fit(X_train, y_train)
# print_score(svm_clf, X_train, y_train, X_test, y_test, train=True)
# print_score(svm_clf, X_train, y_train, X_test, y_test, train=False)

"""
Principal Component Analysis(PCA)
"""
# 1. Уменьшение линейной размерности через разложение данных по сингулярным значениям
# для проецирования их в пространство с более низкой размерностью.
# 2. Неконтролируемое машинное обучение.
# 3. Поиск какие фун-ции объясняют наибольшую дисперсию данных.

# PCA Visualization
# Масштабировать данные, чтобы каждая функция имела дисперсию на 1.
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
scaler = StandardScaler()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

plt.figure(figsize=(8,6))
plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.show()
