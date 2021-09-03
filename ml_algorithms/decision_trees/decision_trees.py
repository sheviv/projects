"""
Random Forest
"""
# Ансамблевый алгоритм машинного обучения(Bootstrap Aggregation или bagging)
# Decision Tree & Random Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# Check data
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

"""
VISUALIZING THE DATA
"""
df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis="columns", inplace=True)
categorical_col = []
for column in df.columns:
    if df[column].dtype == object and len(df[column].unique()) <= 50:
        categorical_col.append(column)
df['Attrition'] = df.Attrition.astype("category").cat.codes

"""
Data Processing
"""
categorical_col.remove('Attrition')

# Преобразование категориальных данных в фиктивные
# data = pd.get_dummies(df, columns=categorical_col)
# data.info()
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
for column in categorical_col:
    df[column] = label.fit_transform(df[column])
from sklearn.model_selection import train_test_split
X = df.drop('Attrition', axis=1)
y = df.Attrition
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

"""
Applying Tree & Random Forest algorithms
"""
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

from sklearn.tree import DecisionTreeClassifier
# tree_clf = DecisionTreeClassifier(random_state=42)
# tree_clf.fit(X_train, y_train)
# print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
# print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)

"""
Decision Tree Classifier Hyper parameter tuning
"""
# Настройка гиперпараметров классификатора дерева решений(выбор нужных параметров из предлагаемых алгоритму)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

params = {
    "criterion":("gini", "entropy"),
    "splitter":("best", "random"),
    "max_depth":(list(range(1, 20))),
    "min_samples_split":[2, 3, 4],
    "min_samples_leaf":list(range(1, 20)),
}
# tree_clf = DecisionTreeClassifier(random_state=42)
# tree_cv = GridSearchCV(tree_clf, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=3)
# tree_cv.fit(X_train, y_train)
# best_params = tree_cv.best_params_
# print(f"Best paramters: {best_params})")
# tree_clf = DecisionTreeClassifier(**best_params)
# tree_clf.fit(X_train, y_train)
# print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
# print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)

# Visualization of a tree
from IPython.display import Image
from six import StringIO
from graphviz import Source
from sklearn import tree
from sklearn.tree import export_graphviz
import pydot
# features = list(df.columns)
# features.remove("Attrition")
# dot_data = StringIO()
# tree.export_graphviz(tree_clf, out_file=dot_data, feature_names=features, filled=True)
# graph = Source(tree.export_graphviz(tree_clf, out_file=dot_data, feature_names=features, filled=True))
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# Image(graph[0].create_png())
# graph.format = 'png'
# graph.render('dtree_render',view=True)

"""
Random Forest
"""
# Метаоценка, соответствует ряду классификаторов дерева решений на различных подвыборках набора данных
# и усредненяет для повышения точности прогнозирования и контроля избыточной подгонки.
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)

"""
Random Forest hyper parameter tuning
"""
# Randomized Search Cross Validation
# Перекрестная проверка
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
# n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# max_features = ['auto', 'sqrt']
# max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
# max_depth.append(None)
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
# bootstrap = [True, False]
#
# random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
#                'max_depth': max_depth, 'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}
# rf_clf = RandomForestClassifier(random_state=42)
# rf_cv = RandomizedSearchCV(estimator=rf_clf, scoring='f1',
#                            param_distributions=random_grid, n_iter=100,
#                            cv=3, verbose=2, random_state=42, n_jobs=-1)
# rf_cv.fit(X_train, y_train)
# rf_best_params = rf_cv.best_params_
# print(f"Best parameters: {rf_best_params})")
# rf_clf = RandomForestClassifier(**rf_best_params)
# rf_clf.fit(X_train, y_train)
# print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
# print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)

# Случайный поиск сузил диапазон для каждого гиперпараметра(можно явно указать каждую комбинацию настроек - GridSearchCV,
# - вместо случайной выборки из распределения оцениваются все комбинации, которые задаются.

# Grid Search Cross Validation
# Подбор лучшей комбинации параметров для алгоритма
n_estimators = [100, 500, 1000, 1500]
max_features = ['auto', 'sqrt']
max_depth = [2, 3, 5]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4, 10]
bootstrap = [True, False]

params_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

rf_clf = RandomForestClassifier(random_state=42)
rf_cv = GridSearchCV(rf_clf, params_grid, scoring="f1", cv=3, verbose=2, n_jobs=-1)
rf_cv.fit(X_train, y_train)
best_params = rf_cv.best_params_
print(f"Best parameters: {best_params}")
rf_clf = RandomForestClassifier(**best_params)
rf_clf.fit(X_train, y_train)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)
# Вывод: Best parameters: {'bootstrap': False, 'max_depth': None, 'max_features': 'auto',
# 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1500}
