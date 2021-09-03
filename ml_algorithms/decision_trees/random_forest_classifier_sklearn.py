import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# загрузка датасета
iris = datasets.load_iris()
X = iris.data[:, 2:]
y = iris.target
# разделение на тестовый и тренировочный набор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1337, stratify=y)
# определение алгоритма Random Forest Classifier
forest = RandomForestClassifier(criterion='gini',
                                n_estimators=5,
                                random_state=1337,
                                n_jobs=2)
# обучение модели
forest.fit(X_train, y_train)
# прогнозирование значений тестовых данных
y_pred = forest.predict(X_test)
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
# plot_decision_regions function takes "forest" as classifier
fig, ax = plt.subplots(figsize=(7, 7))
plot_decision_regions(X_combined, y_combined, clf=forest)
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()