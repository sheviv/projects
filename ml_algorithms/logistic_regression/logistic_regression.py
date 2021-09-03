import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# Check data
data = pd.read_csv("advertising.csv")
# data.head()
# data.info()
# data.describe()

# Exploratory Data Analysis
# plt.figure(figsize=(10, 8))
# data.Age.hist(bins=data.Age.nunique())
# plt.xlabel('Age')

# Графики с различными данными
# sns.jointplot(data["Area Income"], data.Age)

# sns.jointplot(data["Daily Time Spent on Site"], data.Age, kind='kde')

# sns.jointplot(data["Daily Time Spent on Site"], data["Daily Intx = np.linspace(-6, 6, num=1000)ernet Usage"])

# sns.pairplot(data, hue='Clicked on Ad')

# plt.figure(figsize=(12, 8))
# sns.heatmap(data.corr(), annot=True)

# plt.show()

"""
Theory Behind Logistic Regression
"""
# Алгоритм линейной классификации для двухклассовых задач.
# S-образная кривая, принимающая любое действительное число и преобразовывающая в значение от 0 до 1.
# Изображение Лог.регрессии
x = np.linspace(-6, 6, num=1000)
# plt.figure(figsize=(10, 6))
# plt.plot(x, (1 / (1 + np.exp(-x))))
# plt.title("Sigmoid Function")
# plt.show()

# Вывод метрик
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

# ML preprocessing(предобработка данных)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
X = data.drop(['Timestamp', 'Clicked on Ad', 'Ad Topic Line', 'Country', 'City'], axis=1)
y = data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# cat_columns = []
num_columns = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']
ct = make_column_transformer(
    (MinMaxScaler(), num_columns),
    (StandardScaler(), num_columns),
    remainder='passthrough')
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

"""
Implementing Logistic Regression in Scikit-Learn
"""
# Реализация логистической регрессии в Scikit-Learn
from sklearn.linear_model import LogisticRegression
# lr_clf = LogisticRegression(solver='liblinear')
# lr_clf.fit(X_train, y_train)
# print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
# print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)

from sklearn.ensemble import RandomForestClassifier
# rf_clf = RandomForestClassifier(n_estimators=1000)
# rf_clf.fit(X_train, y_train)
# print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
# print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)

"""
Performance Measurement
"""
# 1. Confusion Matrix(Матрица неточностей)
# 2. Precision(Точность(измеряет точность положительных прогнозов))
# 3. Recall(Соотношение положительных экземпляров, правильно обнаруженным)
# 4. F1 Score(среднее гармоническое значение Precision и Recall)
# 5. Precision / Recall Tradeoff(Повышение точности снижает Precision и наоборот)
from sklearn.metrics import precision_recall_curve
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.title("Precisions/recalls tradeoff")

# Выбор порогового значения(дает наилучший компромисс между precision и recall)
# Может потребоваться более высокая точность (точность положительных прогнозов).
# precisions, recalls, thresholds = precision_recall_curve(y_test, lr_clf.predict(X_test))
# plt.figure(figsize=(15, 8))
# plt.subplot(2, 2, 1)
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.subplot(2, 2, 2)
# plt.plot(precisions, recalls)
# plt.xlabel("Precision")
# plt.ylabel("Recall")
# plt.title("PR Curve: precisions/recalls tradeoff")
# plt.show()

# ROC
# Кривая ROC отображает частоту истинных положительных результатов против частоты ложных срабатываний.
# Частота ложных срабатываний(FPR) - отношение отрицательных случаев, которые неправильно классифицируются как положительные.
from sklearn.metrics import roc_curve
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
# fpr, tpr, thresholds = roc_curve(y_test, lr_clf.predict(X_test))
# plt.figure(figsize=(12,8))
# plot_roc_curve(fpr, tpr)
# plt.show()

from sklearn.metrics import roc_auc_score
# print(roc_auc_score(y_test, lr_clf.predict(X_test)))

# Кривая PR - когда положительный класс встречается редко или когда больше нужны ложные срабатывания, чем ложноотрицательные.
# Кривая ROC - когда отрицательный класс встречается редко или когда больше нужны ложные отрицательные результаты, чем ложные срабатывания.

"""
Logistic Regression Hyperparameter tuning
"""
# Настройка гиперпараметров логистической регрессии
from sklearn.model_selection import GridSearchCV
lr_clf = LogisticRegression()
penalty = ['l1', 'l2']
C = [0.5, 0.6, 0.7, 0.8]
class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]
solver = ['liblinear', 'saga']
param_grid = dict(penalty=penalty, C=C, class_weight=class_weight, solver=solver)
lr_cv = GridSearchCV(estimator=lr_clf, param_grid=param_grid, scoring='accuracy',
                    verbose=1, n_jobs=-1, cv=10, iid=True)
lr_cv.fit(X_train, y_train)
best_params = lr_cv.best_params_
print(f"Best parameters: {best_params}")
lr_clf = LogisticRegression(**best_params)
lr_clf.fit(X_train, y_train)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)