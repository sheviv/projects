import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#
#        -> 2.0              -> 0.65
# Linear -> 1.0  -> Softmax  -> 0.25   -> CrossEntropy(y, y_hat)
#        -> 0.1              -> 0.1
#     scores(logits)      probabilities
#                           sum = 1.0
#

# Softmax - экспоненциальная функция к каждому элементу и нормализует
# Разделяет на сумму всех этих экспонент
# Сжимает результат в диапазоне от 0 до 1 - вероятность
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpy:', outputs)

x = torch.tensor([2.0, 1.0, 0.1])
# по значениям по первой оси
outputs = torch.softmax(x, dim=0)
print('softmax torch:', outputs)


# Перекрестная энтропия
# Кросс-энтропийная - измеряет эффективность модели классификации
# Выходом - является значение вероятности от 0 до 1.
# Потеря увеличивается, поскольку прогнозируемая вероятность отклоняется от фактической метки.
def cross_entropy(actual, predicted):
    EPS = 1e-15
    predicted = np.clip(predicted, EPS, 1 - EPS)
    loss = -np.sum(actual * np.log(predicted))
    return loss  # / float(predicted.shape[0])


# y должен быть одним кодированием(пренадлежность к одному классу)
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
Y = np.array([1, 0, 0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

# CrossEntropyLoss в PyTorch(применяется Softmax)
# nn.LogSoftmax + nn.NLLLoss
# NLLLoss = потеря вероятности отрицательного журнала
loss = nn.CrossEntropyLoss()

# У(целевые значения) имеют размер nSamples=1
# Каждый элемент имеет метку класса: 0,1,2
# Y(=target) содержит метки классов, а не вероятности
Y = torch.tensor([0])

# вход имеет размер nSamples x nClasses = 1 x 3
# y_pred(= input) должен быть необработанным, ненормализует оценки(логиты) для каждого класса, а не softmax
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f'PyTorch Loss1: {l1.item():.4f}')
print(f'PyTorch Loss2: {l2.item():.4f}')

# Прогнозы
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(f'Actual class: {Y.item()}, Y_pred1: {predictions1.item()}, Y_pred2: {predictions2.item()}')

# Допустима потеря в батче для нескольких образцов
# Целевое значение имеет размер nBatch=3
# Каждый элемент имеет метку класса: 0,1,2
Y = torch.tensor([2, 0, 1])

# Вход имеет размер nBatch x nClasses = 3 x 3
# Y_pred - логиты(не softmax/вероятности)
Y_pred_good = torch.tensor(
    [[0.1, 0.2, 3.9],  # predict class 2
     [1.2, 0.1, 0.3],  # predict class 0
     [0.3, 2.2, 0.2]])  # predict class 1

Y_pred_bad = torch.tensor(
    [[0.9, 0.2, 0.1],
     [0.1, 0.3, 1.5],
     [1.2, 0.2, 0.5]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f'Batch Loss1:  {l1.item():.4f}')
print(f'Batch Loss2: {l2.item():.4f}')

# Прогнозы
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(f'Actual class: {Y}, Y_pred1: {predictions1}, Y_pred2: {predictions2}')


# Бинарная классификация
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Создание снн сети
        """
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # Сигмоид фун-ия всегда в конце
        y_pred = torch.sigmoid(out)
        return y_pred
model = NeuralNet1(input_size=28 * 28, hidden_size=5)
criterion = nn.BCELoss()


# Мультиклассовая классификация
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out
model = NeuralNet2(input_size=28 * 28, hidden_size=5, num_classes=3)
# (Softmax)
criterion = nn.CrossEntropyLoss()


"""
activation_functions
"""
x = torch.tensor([-1.0, 1.0, 2.0, 3.0])

# Виды функции активации
# sofmax
output = torch.softmax(x, dim=0)
print(output)
sm = nn.Softmax(dim=0)
output = sm(x)
print(output)

# sigmoid
output = torch.sigmoid(x)
print(output)
s = nn.Sigmoid()
output = s(x)
print(output)

# tanh
output = torch.tanh(x)
print(output)
t = nn.Tanh()
output = t(x)
print(output)

# relu
output = torch.relu(x)
print(output)
relu = nn.ReLU()
output = relu(x)
print(output)

# leaky relu
output = F.leaky_relu(x)
print(output)
lrelu = nn.LeakyReLU()
output = lrelu(x)
print(output)


# nn.ReLU() - создает модуль nn.Module, можно добавить к Скрытому слою сети.
# Вариант_1 (создать модуль с "сеткой")
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# Вариант_1 (использовать функции активации непосредственно в прямом проходе)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out


"""
all_plot_functions_activation
"""
import matplotlib.pyplot as plt

##### Sigmoid
sigmoid = lambda x: 1 / (1 + np.exp(-x))
x = np.linspace(-10,10,10)
y = np.linspace(-10,10,100)
fig = plt.figure()
plt.plot(y,sigmoid(y),'b', label='linspace(-10,10,100)')
plt.grid(linestyle='--')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Sigmoid Function')
plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
plt.yticks([-2, -1, 0, 1, 2])
plt.ylim(-2, 2)
plt.xlim(-4, 4)
plt.show()
#plt.savefig('sigmoid.png')
fig = plt.figure()

##### TanH
tanh = lambda x: 2*sigmoid(2*x)-1
x = np.linspace(-10,10,10)
y = np.linspace(-10,10,100)
plt.plot(y,tanh(y),'b', label='linspace(-10,10,100)')
plt.grid(linestyle='--')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('TanH Function')
plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
plt.yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
plt.ylim(-4, 4)
plt.xlim(-4, 4)
plt.show()
#plt.savefig('tanh.png')
fig = plt.figure()

##### ReLU
relu = lambda x: np.where(x>=0, x, 0)
x = np.linspace(-10,10,10)
y = np.linspace(-10,10,1000)
plt.plot(y,relu(y),'b', label='linspace(-10,10,100)')
plt.grid(linestyle='--')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('ReLU')
plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
plt.yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
plt.ylim(-4, 4)
plt.xlim(-4, 4)
plt.show()
#plt.savefig('relu.png')
fig = plt.figure()

##### Leaky ReLU
leakyrelu = lambda x: np.where(x>=0, x, 0.1*x)
x = np.linspace(-10,10,10)
y = np.linspace(-10,10,1000)
plt.plot(y,leakyrelu(y),'b', label='linspace(-10,10,100)')
plt.grid(linestyle='--')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Leaky ReLU')
plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
plt.yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
plt.ylim(-4, 4)
plt.xlim(-4, 4)
plt.show()
#plt.savefig('lrelu.png')
fig = plt.figure()


##### Binary Step
bstep = lambda x: np.where(x>=0, 1, 0)
x = np.linspace(-10,10,10)
y = np.linspace(-10,10,1000)
plt.plot(y,bstep(y),'b', label='linspace(-10,10,100)')
plt.grid(linestyle='--')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Step Function')
plt.xticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
plt.yticks([-2, -1, 0, 1, 2])
plt.ylim(-2, 2)
plt.xlim(-4, 4)
plt.show()
#plt.savefig('step.png')
print('done')
