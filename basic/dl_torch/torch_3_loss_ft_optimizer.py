# 1) Модель проектирования(ввод, вывод, прямой проход с разными слоями)
# 2) Построение фун-ии потерь и оптимизатор
# 3) Цикл обучения
# - forward = вычислить прогноз и потерю
# - backward = вычислить градиенты
# - update weights

import torch
import torch.nn as nn
# f = w * x
# here : f = 2 * x
X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# 1) Модель проектирования: веса для оптимизации
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
def forward(x):
    return w * x
print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

# 2) Вычислить потерю и оптимизатор
learning_rate = 0.01
n_iters = 100
# Фун-ия ошибок и оптимизатор
loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)

# 3) Цикл обучения
for epoch in range(n_iters):
    y_predicted = forward(X)
    l = loss(Y, y_predicted)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 10 == 0:
        print('epoch ', epoch+1, ': w = ', w, ' loss = ', l)
print(f'Prediction after training: f(5) = {forward(5).item():.3f}')

"""
model loss and optimization
"""
# 1) Модель проектирования(ввод, вывод, прямой проход с разными слоями)
# 2) Построение фун-ии потерь и оптимизатор
# 3) Цикл обучения
# - forward = вычислить прогноз и потерю
# - backward = вычислить градиенты
# - update weights

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
n_samples, n_features = X.shape
print(f'#samples: {n_samples}, #features: {n_features}')
# 0) Тестовый образец
X_test = torch.tensor([5], dtype=torch.float32)

# 1) Дизайн модели(должна реализовывать прямой проход)
# Встроенная модель от PyTorch
input_size = n_features
output_size = n_features

# вызов модели с Х
model = nn.Linear(input_size, output_size)

'''
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define diferent layers
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.lin(x)
model = LinearRegression(input_size, output_size)
'''
print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# 2) Вычисление потери и оптимизатора
learning_rate = 0.01
n_iters = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Цикл обучения
for epoch in range(n_iters):
    # предсказание с использованием модели
    y_predicted = model(X)
    l = loss(Y, y_predicted)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    # просмотр параметров каждую 10-ую эпоху
    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l)
print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')