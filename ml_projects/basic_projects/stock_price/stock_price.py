import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

sns.set()


apple = pd.read_csv("data/AAPL.csv")
# print(apple)
# print("days for training:", apple.shape)

"""
Визуализация конечных/закрытых акций за все дни
"""
plt.figure(figsize=(10, 4))
plt.title("Apple's Stock Price")
plt.xlabel("Days")
plt.ylabel("Close Price USD ($)")
plt.plot(apple["Close"])
plt.show()

# Значения закрытия акций:
apple = apple[["Close"]]
# print(apple.head())

# Кол-во дней прогнозирования
futureDays = 25

# Новый целевой столбец со сдвигом «X»(futureDays) единиц/дней вверх.
apple["Prediction"] = apple[["Close"]].shift(-futureDays)
# print(apple.head())
# print(apple.tail())

# Создать набор данных(x), преобразовать(numpy) и удалить последние «x»(futureDays) строк/дней.
x = np.array(apple.drop(["Prediction"], 1))[:-futureDays]
# print(x)

# Создать целевой набор данных(y)
y = np.array(apple["Prediction"])[:-futureDays]
# print(y)


"""
Разделение данных для обучения модели
"""
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)


"""
Создание моделей
"""
# Дерево решений
tree = DecisionTreeRegressor().fit(xtrain, ytrain)
# Линейная регрессия
linear = LinearRegression().fit(xtrain, ytrain)

# Получить последние «x»(futureDays) строк(дней)
xfuture = apple.drop(["Prediction"], 1)[:-futureDays]
xfuture = xfuture.tail(futureDays)
xfuture = np.array(xfuture)
# print(xfuture)


"""
Прогнозирование Деревом решений
"""
treePrediction = tree.predict(xfuture)
# print("Decision Tree prediction =", treePrediction)

"""
Визуализация предсказаний Дерева решений
"""
# predictions = treePrediction
# valid = apple[x.shape[0]:]
# valid["Predictions"] = predictions
# plt.figure(figsize=(10, 6))
# plt.title("Apple's Stock Price Prediction Model(Decision Tree Regressor Model)")
# plt.xlabel("Days")
# plt.ylabel("Close Price USD ($)")
# plt.plot(apple["Close"])
# plt.plot(valid[["Close", "Predictions"]])
# plt.legend(["Original", "Valid", "Predictions"])
# plt.show()


"""
Прогнозирование Линейной регрессией
"""
linearPrediction = linear.predict(xfuture)
# print("Linear regression Prediction =", linearPrediction)

"""
Визуализация предсказаний Линейной регрессией
"""
# predictions = linearPrediction
# valid = apple[x.shape[0]:]
# valid["Predictions"] = predictions
# plt.figure(figsize=(10, 6))
# plt.title("Apple's Stock Price Prediction Model(Linear Regression Model)")
# plt.xlabel("Days")
# plt.ylabel("Close Price USD ($)")
# plt.plot(apple["Close"])
# plt.plot(valid[["Close", "Predictions"]])
# plt.legend(["Original", "Valid", "Predictions"])
# plt.show()
