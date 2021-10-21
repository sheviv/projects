# Загрузка датасета.
import numpy as np
from sklearn import datasets
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
import xgboost as xgb


iris = datasets.load_iris()
X = iris.data
y = iris.target

# Разделение датасета на обучающую/тестовую выборку.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Импорт XGBoost и создание необходимых объектов.
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Задание параметров модели.
param = {
   'max_depth': 3,
   'eta': 0.3,
   # 'silent': 1,
   'objective': 'multi:softprob',
   'num_class': 3}
num_round = 20

# Обучение.
bst = xgb.train(param, dtrain, num_round)
preds = bst.predict(dtest)

# Определение качества модели на тестовой выборке.
best_preds = np.asarray([np.argmax(line) for line in preds])
print(precision_score(y_test, best_preds, average='macro'))
