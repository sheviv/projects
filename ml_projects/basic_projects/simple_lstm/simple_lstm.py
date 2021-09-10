import numpy
import matplotlib.pyplot as plt
import pandas
import math
import keras
from keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# print(f"version: {keras.__version__}")

numpy.random.seed(1337)

"""
Работа с данными
"""
# Загрузка данных
dataframe = pandas.read_csv('airline-passengers.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
# Нормализовать данные в наборе от 0 до 1
min_max_changer = MinMaxScaler(feature_range=(0, 1))
dataset = min_max_changer.fit_transform(dataset)

# Разделение данных на обучающие и тестовые
# train_shape = int(len(dataset) * 0.67)
train_shape = int(len(dataset) * 0.7)
test_shape = len(dataset) - train_shape
train, test = dataset[0:train_shape, :], dataset[train_shape:len(dataset), :]
print(len(train), len(test))

"""
LSTM
"""
# Преобразование значений в матрицу набора данных
# def lstm_data(dataset, look_back=1):
def lstm_data(dataset, look_back):
    x_data, y_data = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        x_data.append(a)
        y_data.append(dataset[i + look_back, 0])
    return numpy.array(x_data), numpy.array(y_data)


# Изменение формы данных, перед отправлением в create_dataset()
look_back = 1
train_data_x, train_data_y = lstm_data(train, look_back)
test_data_x, test_data_y = lstm_data(test, look_back)
# Изменение размера входных данных
train_data_x = numpy.reshape(train_data_x, (train_data_x.shape[0], 1, train_data_x.shape[1]))
test_data_x = numpy.reshape(test_data_x, (test_data_x.shape[0], 1, test_data_x.shape[1]))

"""
Создание LSTM модели
"""
model = keras.Sequential()
model.add(layers.LSTM(4, input_shape=(1, look_back)))
model.add(layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_data_x, train_data_y, epochs=100, batch_size=1, verbose=2)  # verbose=2 - вывод в консоль

"""
Создание и визуализация прогнозов
"""
train_data_predict = model.predict(train_data_x)
test_data_predict = model.predict(test_data_x)
# Нормализовать данные в наборе от 0 до 1
train_data_predict = min_max_changer.inverse_transform(train_data_predict)
train_data_y = min_max_changer.inverse_transform([train_data_y])
test_data_predict = min_max_changer.inverse_transform(test_data_predict)
test_data_y = min_max_changer.inverse_transform([test_data_y])

# Вычисление ошибки
def mean_square_error(y_true, y_predict):
    return K.mean(K.square(y_predict - y_true), axis=-1)
train_data_score = mean_square_error(train_data_y[0], train_data_predict[:, 0])
test_data_score = mean_square_error(test_data_y[0], test_data_predict[:, 0])

# Визуализация предсказанного набора данных
visual_train_data_predict = numpy.empty_like(dataset)  # новый массив без инициированных записей
visual_train_data_predict[:, :] = numpy.nan
visual_train_data_predict[look_back:len(train_data_predict) + look_back, :] = train_data_predict

# Визуализация тесового набора данных
visual_test_data_predict = numpy.empty_like(dataset)
visual_test_data_predict[:, :] = numpy.nan
visual_test_data_predict[len(train_data_predict) + (look_back * 2) + 1:len(dataset) - 1, :] = test_data_predict

# Визуализация предсказанных и тестовых данных
plt.plot(min_max_changer.inverse_transform(dataset))
plt.plot(visual_train_data_predict)
plt.plot(visual_test_data_predict)
plt.show()
