import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

df = pd.read_csv("diabetes.csv")

"""
Data visualization
"""
# Visualizing the distribution of the data for every feature
# Распределения данных для каждой функции
# plt.figure(figsize=(20, 20))
# for i, column in enumerate(df.columns, 1):
#     plt.subplot(3, 3, i)
#     df[df["Outcome"] == 0][column].hist(bins=35, color='blue', label='Have Diabetes = NO', alpha=0.6)
#     df[df["Outcome"] == 1][column].hist(bins=35, color='red', label='Have Diabetes = YES', alpha=0.6)
#     plt.legend()
#     plt.xlabel(column)
# plt.show()

"""
Data Pre-Processing
"""
# print(df.columns)
# How many missing zeros are mising in each feature
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                   'DiabetesPedigreeFunction', 'Age']

# for column in feature_columns:
#     print(f"{column} ==> Missing zeros : {len(df.loc[df[column] == 0])}")

from sklearn.impute import SimpleImputer

fill_values = SimpleImputer(missing_values=0, strategy="mean", copy=False)
df[feature_columns] = fill_values.fit_transform(df[feature_columns])
# for column in feature_columns:
#     print(f"{column} ==> Missing zeros : {len(df.loc[df[column] == 0])}")

# Visualizing the distribution of the data for every feature
# plt.figure(figsize=(20, 20))
# for i, column in enumerate(df.columns, 1):
#     plt.subplot(3, 3, i)
#     df[df["Outcome"] == 0][column].hist(bins=35, color='blue', label='Have Diabetes = NO', alpha=0.6)
#     df[df["Outcome"] == 1][column].hist(bins=35, color='red', label='Have Diabetes = YES', alpha=0.6)
#     plt.legend()
#     plt.xlabel(column)
# plt.show()

from sklearn.model_selection import train_test_split
X = df[feature_columns]
y = df.Outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
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


"""
Bagging
"""
scores = {}
# Выбор нескольких выборок из набора обучающих данных(с заменой) и обучение модели для каждой выборки.
# Выходной прогноз усредняется по прогнозам всех подмоделей.

"""
Bagged Decision Trees
"""
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
# tree = DecisionTreeClassifier()
# bagging_clf = BaggingClassifier(base_estimator=tree, n_estimators=1500, random_state=42)
# bagging_clf.fit(X_train, y_train)
# evaluate(bagging_clf, X_train, X_test, y_train, y_test)
# scores = {
#     'Bagging Classifier': {
#         'Train': accuracy_score(y_train, bagging_clf.predict(X_train)),
#         'Test': accuracy_score(y_test, bagging_clf.predict(X_test)),
#     },
# }
# print(f"scores: {scores}")

"""
Random Forest
"""
from sklearn.ensemble import RandomForestClassifier
# rf_clf = RandomForestClassifier(random_state=42, n_estimators=1000)
# rf_clf.fit(X_train, y_train)
# evaluate(rf_clf, X_train, X_test, y_train, y_test)
# scores['Random Forest'] = {
#         'Train': accuracy_score(y_train, rf_clf.predict(X_train)),
#         'Test': accuracy_score(y_test, rf_clf.predict(X_test)),
# }
# print(f"scores: {scores}")

"""
Extra Trees
"""
# Вид Bagging, где случайные деревья строятся из образцов обучающего набора данных.
from sklearn.ensemble import ExtraTreesClassifier
# ex_tree_clf = ExtraTreesClassifier(n_estimators=1000, max_features=7, random_state=42)
# ex_tree_clf.fit(X_train, y_train)
# evaluate(ex_tree_clf, X_train, X_test, y_train, y_test)
# scores['Extra Tree'] = {
#         'Train': accuracy_score(y_train, ex_tree_clf.predict(X_train)),
#         'Test': accuracy_score(y_test, ex_tree_clf.predict(X_test)),
#     }
# print(f"scores: {scores}")

"""
Boosting Algorithms
"""
# Последовательность моделей, которые исправляют ошибки моделей, предшествующих им.

"""
AdaBoost
"""
# Взвешивание экземпляров в наборе данных в зависимости от того, насколько сложно их классифицировать,
# что позволяет алгоритму уделять им или меньше внимания при построении последующих моделей.
from sklearn.ensemble import AdaBoostClassifier
# ada_boost_clf = AdaBoostClassifier(n_estimators=30)
# ada_boost_clf.fit(X_train, y_train)
# evaluate(ada_boost_clf, X_train, X_test, y_train, y_test)
# scores['AdaBoost'] = {
#         'Train': accuracy_score(y_train, ada_boost_clf.predict(X_train)),
#         'Test': accuracy_score(y_test, ada_boost_clf.predict(X_test)),
#     }
# print(scores)

"""
Stochastic Gradient Boosting
"""
# Строит аддитивную модель поэтапно, позволяя оптимизировать произвольные дифф. функции потерь на каждом этапе.
from sklearn.ensemble import GradientBoostingClassifier
# grad_boost_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
# grad_boost_clf.fit(X_train, y_train)
# evaluate(grad_boost_clf, X_train, X_test, y_train, y_test)
# scores['Gradient Boosting'] = {
#         'Train': accuracy_score(y_train, grad_boost_clf.predict(X_train)),
#         'Test': accuracy_score(y_test, grad_boost_clf.predict(X_test)),
#     }

"""
Voting Ensemble
"""
# Создает две или более моделей. Использует "голосование", чтобы обернуть модели и усреднить прогнозы подмоделей.
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
estimators = []
log_reg = LogisticRegression(solver='liblinear')
estimators.append(('Logistic', log_reg))
tree = DecisionTreeClassifier()
estimators.append(('Tree', tree))
svm_clf = SVC(gamma='scale')
estimators.append(('SVM', svm_clf))
voting = VotingClassifier(estimators=estimators)
voting.fit(X_train, y_train)
evaluate(voting, X_train, X_test, y_train, y_test)

scores['Voting'] = {
        'Train': accuracy_score(y_train, voting.predict(X_train)),
        'Test': accuracy_score(y_test, voting.predict(X_test)),
    }
scores_df = pd.DataFrame(scores)
scores_df.plot(kind='barh', figsize=(15, 8))
plt.show()
