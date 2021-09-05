import torch
import numpy as np

x = torch.tensor(1.0)
y = torch.tensor(2.0)

# Это параметр, который нужно оптимизировать -> requires_grad=True
w = torch.tensor(1.0, requires_grad=True)

# "прямой проход" для вычисление ошибки
y_predicted = w * x
loss = (y_predicted - y)**2
print(loss)

# "обратный проход" для вычисления градиента dLoss/dw
loss.backward()
print(w.grad)

# Обновление весов
# Следующий проход "вперед и назад"
# Продолжить оптимизацию:
# Обновить веса(операция не должна быть частью вычислительного графа)
with torch.no_grad():
    w -= 0.01 * w.grad
# обнулить градиенты для следующего шага
w.grad.zero_()

"""
gradient descent manually
"""
# Вычисляйте каждый шаг вручную
# Линейная регрессия
# f = w * x
# here : f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)
w = 0.0
# Функция умножения весов на значения
def forward(x):
    return w * x

# функция ошибок: loss = MSE
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

# J = MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)
def gradient(x, y, y_pred):
    return np.dot(2 * x, y_pred - y).mean()
print(f'Prediction before training: f(5) = {forward(5):.3f}')

# Обучение модели
learning_rate = 0.01
n_iters = 20
for epoch in range(n_iters):
    # предсказание = вычисление значения * веса
    y_pred = forward(X)
    # ошибка
    l = loss(Y, y_pred)
    # вычисление градиента
    dw = gradient(X, Y, y_pred)
    # обновление весов
    w -= learning_rate * dw
    if epoch % 2 == 0:
        print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')
print(f'Prediction after training: f(5) = {forward(5):.3f}')

"""
autograd
"""
# Замена вычисленного вручную градиента на autograd.
# Линейная регрессия
# f = w * x
# here : f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# Функция умножения весов на значения
def forward(x):
    return w * x

def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

print(f'Prediction before training: f(5) = {forward(5).item():.3f}')
learning_rate = 0.01
n_iters = 100
for epoch in range(n_iters):
    y_pred = forward(X)
    l = loss(Y, y_pred)
    l.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
    # обнулить градиенты после обновления
    w.grad.zero_()
    if epoch % 10 == 0:
        print(f'epoch {epoch + 1}: w = {w.item():.3f}, loss = {l.item():.8f}')
print(f'Prediction after training: f(5) = {forward(5).item():.3f}')
