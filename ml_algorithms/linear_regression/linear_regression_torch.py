import numpy as np
import matplotlib.pyplot as plt
import torch

# x_data = np.array([1, 2, 3, 4, 5])
# y_data = x_data * 2
# def forward(x, w, b=0):
#     y_hat = x * w + b
#     return y_hat
# def loss(y_hat, y):
#     return (y_hat - y) ** 2
# all_w = []
# all_loss = []
# for w in np.arange(0, 4, 0.1):
#     l_sum = 0
#     for i in range(len(x_data)):
#         y_hat = forward(x_data[i], w)
#         l = loss(y_hat, y_data[i])
#         l_sum += l
#     all_w.append(w)
#     all_loss.append(l_sum / len(y_data))
# x_torch = torch.FloatTensor(x_data).reshape(-1, 1)
# y_torch = torch.FloatTensor(y_data).reshape(-1, 1)
# w = torch.tensor(5., requires_grad=True)
# b = torch.tensor(3., requires_grad=True)
# lr = 0.05
# for i in range(1000):
#     y_hat = x_torch * w + b
#     loss = torch.sum(torch.pow(y_torch - y_hat, 2) / len(y_torch))
#     loss.backward()
#     with torch.no_grad():
#         w -= lr * w.grad
#         b -= lr * b.grad
#         w.grad.zero_()
#         b.grad.zero_()
# y_pred = x_torch * w + b
# print(f"y_pred: {y_pred}")
# plt.plot(x_torch, y_torch, 'go')
# plt.plot(x_torch, y_pred.detach().numpy(), '--')
# plt.show()


# # генерируются случайным образом в интервале от 0 до 1 50 чисел
# x = np.random.rand(50)
# # растянуть значения x до значений в интервале от 0 до 10, умножив x на 10
# x = x * 10
# y = x * 3 - 4
# # добавление шума к y
# y += np.random.randn(50)
# # модель PyTorch, torch.nn.Module - параметр
# class LinearModel(torch.nn.Module):
#     """
#     два параметра:
#     1 - размер каждой входной выборки(=1 т.к. только числа)
#     2 - форма вывода(=1)
#     """
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.linear = torch.nn.Linear(1, 1)
#     def forward(self, x):
#         return self.linear(x)
# # преобразование данных в тензоры
# x_torch = torch.FloatTensor(x).reshape(-1, 1)
# y_torch = torch.FloatTensor(y).reshape(-1, 1)
# # рассчет потери с помощью функции torch.nn.MSELoss()
# model = LinearModel()
# # рассчет среднеквадратичной ошибки
# criterion = torch.nn.MSELoss()
# # SGD(стохастический градиентный спуск) для вычисления градиентов
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# all_loss = []
# for epoch in range(1000):
#     # критерий прогнозируемого значения
#     y_hat = model(x_torch)
#     loss = criterion(y_hat, y_torch)
#     loss.backward()
#     all_loss.append(loss.item())
#     # вычисление градиентов(обратное распространение)
#     optimizer.step()
#     # убедиться, что вычисленный градиент =0 после каждой эпохи
#     optimizer.zero_grad()
# # прогноз значений
# y_pred = model.forward(x_torch)
# # рассчитанные значения веса и смещения
# for name, param in model.named_parameters():
#     print(name, param)
# plt.plot(np.arange(0, 200, 1), all_loss[:200])
# plt.show()
# print(f"all_loss: {all_loss[-1]}")

# ////////////////////////////////////////

import torch
from torch.autograd import Variable
# x_data - независимая переменная
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
# y_data - зависимая переменная
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        """
        Model является подклассом torch.nn.module
        """
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # Один вход, один выход
    def forward(self, x):
        """
        прямое распространение
        :param x:
        :return:
        """
        y_pred = self.linear(x)
        return y_pred
# our model
our_model = LinearRegressionModel()
# оптимизатор и критерии потерь
# среднеквадратичная ошибка(MSE)
criterion = torch.nn.MSELoss(size_average=False)
# стохастический градиентный спуск (SGD) с шагом обучения 0.01
optimizer = torch.optim.SGD(our_model.parameters(), lr=0.01)
# кол-во эпох 500
for epoch in range(500):
    # прямой проход: вычислить предсказанный y, передав x модели
    pred_y = our_model(x_data)
    # вычислить потерю
    loss = criterion(pred_y, y_data)
    # обнулить градиенты
    optimizer.zero_grad()
    # выполнить обратный проход
    loss.backward()
    # обновить веса
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))
new_var = Variable(torch.Tensor([[4.0]]))
# предсказание значения new_var по моделе
pred_y = our_model(new_var)
print("predict (after training)", 4, our_model(new_var).item())