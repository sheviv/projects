import numpy as np
import tensorflow as tf
# преобразуются образцы из целых чисел в числа с плавающей запятой
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# оптимизатор и функция потерь для обучения
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
# модель возвращает вектор оценок «логитов» или «логарифмов шансов», по одному для каждого класса
predictions = model(x_train[:1]).numpy()
# преобразование логитов в "вероятности" для каждого класса
soft_pred = tf.nn.softmax(predictions).numpy()
# вектор логитов и индекс True и возвращает скалярную потерю для каждого примера
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# потеря равна отрицательной логарифмической вероятности истинного класса
loss_fn_rand = loss_fn(y_train[:1], predictions).numpy()
# сборка модели с параметрами
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
# регулировка параметров модели для минимизации потери
model.fit(x_train, y_train, epochs=5)
# проверка производительности модели на «наборе проверки» или «наборе тестов»
model.evaluate(x_test,  y_test, verbose=2)
# модель возвращает вероятность(обернуть модель и прикрепить softmax)
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()])
# print(probability_model(x_test[:5]))
