import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

"""
Аугментация и разделение данных для обучения и тестирования
"""
train_data_generation = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=40,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           fill_mode='nearest')
test_data_generation = ImageDataGenerator(rescale=1.0 / 255)
train_generator = train_data_generation.flow_from_directory("path/Train/",
                                                            batch_size=256,
                                                            class_mode='binary',
                                                            target_size=(64, 64))
validation_generator = test_data_generation.flow_from_directory("path/Validation/",
                                                                batch_size=256,
                                                                class_mode='binary',
                                                                target_size=(64, 64))

"""
Создание, компиляция и обучение модели сети
"""
# Dropout() / SpatialDropout2D() - после слоя пулинга
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),

    tf.keras.layers.Conv2D(256, (11, 11), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(
    optimizer=Adam(lr=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
hist = model.fit_generator(generator=train_generator,
                           validation_data=validation_generator,
                           steps_per_epoch=256,
                           validation_steps=256,
                           epochs=50)

"""
Визуализация точности модели
"""
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

"""
Тестирование модели(сети) на новых данных
"""
path = "image.jpg"
img = image.load_img(path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])  # соединение массивов по вертикали
classes = model.predict(images, batch_size=1)
print(classes[0])
if classes[0] > 0.5:
    print("man")
else:
    print("female")
plt.imshow(img)
