from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
# классификатор опорных векторов
from sklearn.svm import SVC
from sklearn import metrics

# загрузка датасета
iris = datasets.load_iris()
X = iris.data
y = iris.target
# разделение данных на обучающие и проверочные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% обучающих и 30% тестовых
# SVC в качестве базовой оценки
svc = SVC(probability=True, kernel='linear')
# классификатор adaboost
abc = AdaBoostClassifier(n_estimators=50, base_estimator=svc, learning_rate=1)
# обучение Adaboost классификатора
model = abc.fit(X_train, y_train)
# предсказание ответа для тестового набора данных
y_pred = model.predict(X_test)
# точность модели(как часто классификатор верно предсказывает ответы)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
