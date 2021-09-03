import tensorflow as tf
import numpy as np

# Различные способы представления тензоров
# Задает тремя способами матрицу2 х 2
# m1 = [[1.0, 2.0],
#       [3.0, 4.0]]
# m2 = np.array([[1.0, 2.0],
#                [3.0, 4.0]],
#               dtype=np.float32)
# m3 = tf.constant([[1.0, 2.0],
#                   [3.0, 4.0]])

# Создает тензорные объекты одного типа из разных типов данных
# tl = tf.convert_to_tensor(m1, dtype=tf.float32)
# t2 = tf.convert_to_tensor(m2, dtype=tf.float32)
# t3 = tf.convert_to_tensor(m3, dtype=tf.float32)

# Создание тензоров
# m1 = tf.constant([[1., 2.]])
# m2 = tf.constant([[1],
#                  [2]])
# m3 = tf.constant([[[1, 2],
#                    [3, 4],
#                    [5, 6]],
#                   [[7, 8],
#                    [9, 10],
#                    [11, 12]]])

# Использование оператора изменения знака
# x = tf.constant([[1, 2]])
# y = tf.constant([[1, 2]])
# negMatrix = tf.negative(x)  # Меняетзнакэпементовтензора

# ОПЕРАТОРЫ TENSORFLOW
# tf.add(x, y)  # складывает два тензора одного типа, х +у
# tf.subtract(x, y)  # вычитает тензоры одного типа, х-у
# tf.multiply(x, y)  # перемножает поэлементно два тензора
# tf.pow(x, y)  # возводит поэлементно х в степень у
# tf.ехр(x)  # аналогичен pow(e, х), где е - число Эйлера (2,718. ")
# tf.sqrt(x)  # аналогичен pow(x, 0,5)
# tf.div(x, y)  # выполняет поэлементно деление х и у
# tf.truediv(x, y)  # то же самое, что и tf. div, кроме того, что приводит аргументы как числа с плавающей запятой
# tf.floordiv(x, y)  # то же самое, что и truediv, но с округлением окончательного результата до целого числа
# tf.mod(x, y)  # берет поэлементно остаток от деления

# Использование сеанса
# x = tf.constant([[1., 2.]])  # Задает произвольную матрицу
# neg_op = tf.negative(x)  # Выполняет на ней оператор изменения знака
# with tf.Session() as sess:  # Начинаетсеанс, чтобы можно было выполнить операции
#     result = sess.run(negMatrix)  # Сообщает в сеансе о необходимости оценить пegMatrix
# print(result)  # Выводит результирующую матрицу

# Применение интерактивного режима сеанса
# sess = tf.compat.v1.InteractiveSession()  # Начинает интерактивный сеанс(переменную сеанса больше не требуется упоминать)
# x = tf.constant([[1., 2.]])  # Задает произвольную матрицу и изменяет знак ее элементов
# negMatrix = tf.negative(x)
# result = negMatrix.eval()  # Теперь можно вычислить negMatrix, не задавая явным образом сеанс
# print(result)
# sess.close()  # закрыть сеанс, чтобы освободить ресурсы

# Настройка конфигурации сеансов
# Регистрация сеанса
# x = tf.constant([[1., 2.]])
# negMatrix = tf.negative(x)
# with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:
#     result = sess.run(negMatrix)
# print(result)

# Сохранение и загрузка переменных
# sess = tf.compat.v1.InteractiveSession()
# raw_data = [1., 2., 8., -1., 0., 5.5, 6., 13]
# spikes = tf.Variable([False] * len(raw_data), name='spikes')
# spikes.initializer.run()
# saver = tf.compat.v1.train.Saver
# for i in range(1, len(raw_data)):
#     if raw_data[i] - raw_data[i-1] > 5:
#         spikes_val = spikes.eval()
#         spikes_val[i] = True
#         updater = tf.compat.v1.assign(spikes, spikes_val)
#         updater.eval()
# save_path = saver.save(sess, "./spikes.ckpt")
# print("spikes data saved in file: %s" % save_path)
# sess.close()