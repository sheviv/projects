import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder  # for train test splitting
from sklearn.model_selection import train_test_split  # for decision tree object
from sklearn.tree import DecisionTreeClassifier  # for checking testing results
from sklearn.metrics import classification_report, confusion_matrix  # for visualizing tree
from sklearn.tree import plot_tree

# загрузка набора данных
df = sns.load_dataset('iris')
df.isnull().any()
target = df['species']
df1 = df.copy()
df1 = df1.drop('species', axis=1)
# определение атрибутов
X = df1
# сопоставление подписей
le = LabelEncoder()
target = le.fit_transform(target)
y = target
# разделение на тестовый и тренировочный набор(80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# определение алгоритма дерева решений algorithm_dtree=DecisionTreeClassifier()
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
# прогнозирование значений тестовых данных
y_pred = dtree.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
# визуализация и сохранение дерева(графа) plt.figure(figsize=(20, 20))
fig = plt.figure(figsize=(20, 20))
dec_tree = plot_tree(decision_tree=dtree, feature_names=df1.columns,
                     class_names=["setosa", "vercicolor", "verginica"],
                     filled=True, precision=4, rounded=True)
fig.savefig("decistion_tree.png")
