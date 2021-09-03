import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# actual weight = 2 and actual bias = 0.9
# создать синтетические данные для модели
x = np.linspace(0, 3, 120)
# наколн прямой = 2, постоянное значение = 0,9
y = 2 * x + 0.9 + np.random.randn(*x.shape) * 0.3
plt.scatter(x, y, label="input data set")
plt.show()


class LinearModel:
    def __call__(self, x):
        """
        call возвращает значения в соответствии с уравнением прямой линии y=mx+c
        :param x: синтетические данные для модели
        :return:
        """
        return self.Weight * x + self.Bias

    def __init__(self):
        """
        init инициализирует вес и смещение случайным образом
        """
        self.Weight = tf.Variable(11.0)
        self.Bias = tf.Variable(12.0)


def loss(y, pred):
    """
    квадрат разницы y и y' значения, затем tf.reduce_mean для квадратного корня из среднего
    :param y: фактическое значение зависимой переменной
    :param pred: предсказанное значение зависимой переменной
    :return:
    """
    return tf.reduce_mean(tf.square(y - pred))


def train(linear_model, x, y, lr=0.12):
    """
    :param linear_model: экземпляр модели
    :param x: независимая переменная
    :param y: зависимая переменная
    :param lr: скорость обучения
    :return:
    """
    # градиент вычисления относительно его входных переменных
    with tf.GradientTape() as t:
        current_loss = loss(y, linear_model(x))
    lr_weight, lr_bias = t.gradient(current_loss, [linear_model.Weight, linear_model.Bias])
    linear_model.Weight.assign_sub(lr * lr_weight)
    linear_model.Bias.assign_sub(lr * lr_bias)


linear_model = LinearModel()
Weights, Biases = [], []
epochs = 80  # кол-во эпох
for epoch_count in range(epochs):
    Weights.append(linear_model.Weight.numpy())
    Biases.append(linear_model.Bias.numpy())
    real_loss = loss(y, linear_model(x))
    train(linear_model, x, y, lr=0.12)  # lr - шаг обучения
    # рассчитываем потери в каждую эпоху
    # как уменьшается величина потерь по мере увеличения числа эпох
    print(f"Epoch count {epoch_count}: Loss value: {real_loss.numpy()}")


# значения веса и смещения
linear_model_weight, linear_model_bias = linear_model.Weight.numpy(), linear_model.Bias.numpy()
print(f"linear_model: {linear_model_weight}, {linear_model_bias}")
# Среднеквадратическая ошибка(квадратный корень из MSE)
RMSE = loss(y, linear_model(x))
print(f"RMSE.numpy: {RMSE.numpy()}")
