import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
# все изображения должны иметь одинаковые размеры и свойства
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])
# загрузка данных и загрузка(объединяет набор данных)
trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
# просмотр формы изображений и подписей
dataiter = iter(trainloader)
images, labels = dataiter.next()
input_size = 784  # кол-во нейронов во входном слое
hidden_sizes = [128, 64]  # кол-во нейронов в первом и втором скрытом слое
output_size = 10  # кол-во нейронов в выходном слое
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
# логарифмическая вероятность пренадлоежности к классу
criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)
logps = model(images)  # логарифмические вероятности
loss = criterion(logps, labels)  # рассчет NLL потери
# перебор обучающего набора
# оптимизатор
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15  # кол-во эпох
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Сглаживайте изображения MNIST в 784 длинный вектор
        images = images.view(images.shape[0], -1)
        # Пропуск обучения
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        # модель учится путем обратного распространения ошибки
        loss.backward()
        # оптимизация весов
        optimizer.step()
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))
print("\nTraining Time (in minutes) =", (time() - time0) / 60)
images, labels = next(iter(valloader))
img = images[0].view(1, 784)
with torch.no_grad():
    logps = model(img)
ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))