import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd

data_train = pd.read_csv('data/sign_mnist_train.csv')
data_test = pd.read_csv('data/sign_mnist_test.csv')


"""
Разделение данных для обучения и тестирования
"""
# Изменение размерности данных(добавить новый канал - 1(.,.,.,1)) - добавление канала серого
training_images = data_train.iloc[:, 1:].values
training_labels = data_train.iloc[:, 0].values

testing_images = data_test.iloc[:, 1:].values
testing_labels = data_test.iloc[:, 0].values

training_images = training_images.reshape(-1, 28, 28, 1)
testing_images = testing_images.reshape(-1, 28, 28, 1)
# print(training_images.shape) # (27455, 28, 28, 1)
# print(training_labels.shape) # (27455,)
# print(testing_images.shape) # (7172, 28, 28, 1)
# print(testing_labels.shape)  # (7172,)

# Вывести 10 первых изображений
# fig, ax = plt.subplots(2, 5)
# fig.set_size_inches(10, 10)
# k = 0
# for i in range(2):
#     for j in range(5):
#         ax[i, j].imshow(training_images[k].reshape(28, 28), cmap="gray")
#         k += 1
#     plt.tight_layout()
# plt.show()


"""
Аугментация набора данных
"""
# Увеличение набора данных
gen_train_dataset = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
gen_validation_dataset = ImageDataGenerator(rescale=1 / 255)
# print(training_images.shape)  # (27455, 28, 28, 1)
# print(testing_images.shape)  # (7172, 28, 28, 1)


"""
Создание модели нейронной сети
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Dropout(0.4),  # SpatialDropout2D(0.5) - удаляют карты функций
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

# Гиперпараметры
batch_size = 32
epochs = 10
ada_delta = tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name="Adadelta")
sparse_loss = tf.keras.losses.SparseCategoricalCrossentropy()
# acc_metric = tf.keras.metrics.Accuracy()

# model.compile(
#     optimizer=ada_delta,
#     loss=sparse_loss,
#     metrics=['accuracy']
# )
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit_generator(gen_train_dataset.flow(training_images, training_labels, batch_size=batch_size),
                              steps_per_epoch=len(training_images) / batch_size,
                              epochs=epochs,
                              validation_data=gen_validation_dataset.flow(testing_images, testing_labels, batch_size=batch_size),
                              validation_steps=len(testing_images) / batch_size)
model.evaluate(testing_images, testing_labels, verbose=0)


# Виуализация точности и ошибки на моделе
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(len(acc))
# plt.plot(epochs, acc, 'r', label='Training accuracy')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
# plt.title('Training and validation accuracy')
# plt.legend(loc=0)
# plt.figure()

# plt.plot(epochs, loss, 'r', label='Training Loss')
# plt.plot(epochs, val_loss, 'b', label='Validation Loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()

# Предсказания модели на данных
predictions = model.predict_classes(testing_images)
for i in range(len(predictions)):
    if predictions[i] >= 9:
        predictions[i] += 1
print(f"predictions[:5]: {predictions[:5]}")

# Метрики точности
classes = ["Class " + str(i) for i in range(26) if i != 9]
print("Accuracy metrics", classification_report(data_test['label'], predictions, target_names=classes))

# Матрица неточностей для прогнозов кадого класса
cm = confusion_matrix(data_test['label'], predictions)
plt.figure(figsize=(12, 7))
g = sns.heatmap(cm, cmap='Reds', annot=True, fmt='')
plt.show()


# .save_model()
# .load()
