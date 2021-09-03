import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x1_label0 = np.random.normal(1, 1, (100, 1))  #1 Генерируетточки вбпизи (1, 1)
x2_label0 = np.random.normal(1, 1, (100, 1))
x1_label1 = np.random.normal(5, 1, (100, 1))  #1 Генерирует точки вблизи (5, 4)
x2_label1 = np.random.normal(4, 1, (100, 1))
x1_label2 = np.random.normal(8, 1, (100, 1))  #1 Генерирует точки вбпизи (8, О)
x2_label2 = np.random.normal(0, 1, (100, 1))
plt.scatter(x1_label0, x2_label0, c='r', marker='o', s=60)  #Визуапизирует три
plt.scatter(x1_label1, x2_label1, c='g', marker='x', s=60)
plt.scatter(x1_label2, x2_label2, c='b', marker='*', s=60)  #класса на диаграмме
plt.show()
