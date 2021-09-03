from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# загрузка датасета
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")
X = dataset[:, 0:8]
Y = dataset[:, 8]
seed = 7
test_size = 0.33
# разделение данных на обучающие и проверочные
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# обучение Xgboost классификатора
model = XGBClassifier()
model.fit(X_train, y_train)
# предсказание ответа для тестового набора данных
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# оценка точности предсказаний
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
