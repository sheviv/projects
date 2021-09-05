import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# Вычисление градиента неэффективно для всего набора данных -> разделить набор данных на небольшие партии(batches)
'''
# training loop
for epoch in range(num_epochs):
    # loop over all batches
    for i in range(total_batches):
        batch_x, batch_y = ...
'''
# epoch - один прямой и обратный проход всех обучающих выборок
# batch_size - количество обучающих выборок, использованных за один проход
# number of iterations - количество проходов с использованием количества выборок
class WineDataset(Dataset):
    def __init__(self):
        # Инициализация данных
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        # Первый столбец - это метка класса, остальные - функции
        self.x_data = torch.from_numpy(xy[:, 1:])  # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]])  # size [n_samples, 1]

    # Поддержка индексации, так что набор dataset[i] может использоваться для получения i-й выборки
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # Вызвать len(dataset), чтобы вернуть размер
    def __len__(self):
        return self.n_samples

# Создать набор данных
dataset = WineDataset()

# Распаковать первый образец
first_data = dataset[0]
features, labels = first_data
print(features, labels)

# Загрузить весь набор данных через DataLoader()
# shuffle - перемешать данные
# num_workers - более быстрая загрузка с несколькими подпроцессами
# num_workers=0 - если ошибка при загрузке данных
train_loader = DataLoader(dataset=dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=2)

# Преобразовать в итератор и вывести случайную выборку
dataiter = iter(train_loader)
data = dataiter.next()
features, labels = data
print(features, labels)

# Фиктивное тренировочное обучение
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)
print(total_samples, n_iterations)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # here: 178 samples, batch_size = 4, n_iters=178/4=44.5 -> 45 iterations
        # Запустить тренировочный процесс
        if (i + 1) % 5 == 0:
            print(
                f'Epoch: {epoch + 1}/{num_epochs}, Step {i + 1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')
# Пример набора данных из torchvision.datasets
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)
# Загрузить набор данных train_dataset в DataLoader()
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=3,
                          shuffle=True)
# Вывести случайный образец
dataiter = iter(train_loader)
data = dataiter.next()
inputs, targets = data
print(inputs.shape, targets.shape)

"""
transformers
Преобразования можно применять к изображениям PIL, тензорам, ndarrays или пользовательским данным.
"""
class WineDataset(Dataset):
    def __init__(self, transform=None):
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        # Здесь не преобразовывать в тензор
        self.x_data = xy[:, 1:]
        self.y_data = xy[:, [0]]
        self.transform = transform

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

# Преобразования
class ToTensor:
    # Преобразовать в Tensors
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    # Умножать входы с заданным коэффициентом
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

# Показать данные с/без преобразованиями данных
print('Without Transform')
dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nWith Tensor Transform')
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nWith Tensor and Multiplication Transform')
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)
