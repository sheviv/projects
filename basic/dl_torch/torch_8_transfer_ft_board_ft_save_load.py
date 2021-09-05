import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(class_names)


def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()


# Batch обучающих данных
inputs, classes = next(iter(dataloaders['train']))
# Сделать сетку из batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # У каждой эпохи есть этап обучения и проверки.
        for phase in ['train', 'val']:
            if phase == 'train':
                # Установить модель в режим обучения
                model.train()
            else:
                # Установить модель в режим оценки
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Перебор свех данных
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Прмой проход
                # Отслеживать историю обучения
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Обратный проход + оптимизация, только в фазе обучения
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Статистика
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Глубокая копия модели, для сохранения лучших показателей
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Загрузить лучшие веса модели
    model.load_state_dict(best_model_wts)
    return model


"""
Тонкая настройка свёрточной сети
"""
# Загрузка предварительно обученной модели(.resnet18) и сброс полносвязного слой.
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
# Размер каждого выходного примера равен 2.
# Можно обобщить до nn.Linear(num_ftrs, len(class_names)).
model.fc = nn.Linear(num_ftrs, 2)

model = model.to(device)
criterion = nn.CrossEntropyLoss()

# Оптимизировать все параметры модели
optimizer = optim.SGD(model.parameters(), lr=0.001)

# StepLR - уменьшение скорости обучения каждой группы параметров по гамме за каждую эпоху step_size
# Уменьшение LR с коэффициентом 0,1 каждые 7 эпох
# Планирование скорости обучения должно применяться после обновления оптимизатора
# for epoch in range(100):
#     train(...)
#     validate(...)
#     scheduler.step()

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=25)

"""
ConvNet как средство извлечения фиксированных функций
"""
# Заморозка всей сети, кроме последнего слоя.
# requires_grad==False - заморозить параметры, чтобы градиенты не вычислялись в backward()
model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

# Параметры новых построенных модулей по умолчанию имеют required_grad=True.
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)
model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()

# Оптимизируются только параметры последнего слоя, в отличие от предыдущего.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Уменьшение LR в 0,1 раза каждые 7 эпох
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)


"""
tensorboard
"""
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# TENSORBOARD
writer = SummaryWriter('runs/mnist1')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784  # 28x28
hidden_size = 500
num_classes = 10
num_epochs = 1
batch_size = 64
learning_rate = 0.001

train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

examples = iter(test_loader)
example_data, example_targets = examples.next()

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(example_data[i][0], cmap='gray')

# TENSORBOARD
img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('mnist_images', img_grid)
# writer.close()
# sys.exit()
#

# Полносвязная нейронная сеть с одним скрытым слоем
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# TENSORBOARD
writer.add_graph(model, example_data.reshape(-1, 28 * 28))
# writer.close()
# sys.exit()
#

# Обучении модели
running_loss = 0.0
running_correct = 0
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
            # TENSORBOARD
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
            running_accuracy = running_correct / 100 / predicted.size(0)
            writer.add_scalar('accuracy', running_accuracy, epoch * n_total_steps + i)
            running_correct = 0
            running_loss = 0.0
            #

class_labels = []
class_preds = []
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        values, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        class_probs_batch = [F.softmax(output, dim=0) for output in outputs]

        class_preds.append(class_probs_batch)
        class_labels.append(predicted)

    # 10000, 10, and 10000, 1
    # stack - объединяет тензоры в новом измерении
    # cat - объединяет тензоры в заданном измерении
    class_preds = torch.cat([torch.stack(batch) for batch in class_preds])
    class_labels = torch.cat(class_labels)

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

    # TENSORBOARD
    classes = range(10)
    for i in classes:
        labels_i = class_labels == i
        preds_i = class_preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()
    #

"""
save_load
"""
""" 
Сохранение

3 метода:
 - torch.save(arg, PATH) # can be model, tensor, or dictionary
 - torch.load(PATH)
 - torch.load_state_dict(arg)
"""

"""
2 вида сохранения:

1) Сохранить всю модель
torch.save(model, PATH)
# класс модели должен быть где-то определен
model = torch.load(PATH)
model.eval()
2) Сохранить только state_dict
torch.save(model.state_dict(), PATH)
# модель должна быть создана заново с параметрами
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
"""

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
model = Model(n_input_features=6)
# Далее обучение модели ...
# save all
for param in model.parameters():
    print(param)

# Сохранить и загрузить всю модель
FILE = "model.pth"
torch.save(model, FILE)
loaded_model = torch.load(FILE)
loaded_model.eval()
for param in loaded_model.parameters():
    print(param)

# Сохранить только состояние в словарь
FILE = "model.pth"
torch.save(model.state_dict(), FILE)
print(model.state_dict())
loaded_model = Model(n_input_features=6)
# Берется загруженный словарь, а не сам файл пути
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()
print(loaded_model.state_dict())


# Загрузка контрольных точек
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict()
}
print(optimizer.state_dict())
FILE = "checkpoint.pth"
torch.save(checkpoint, FILE)

model = Model(n_input_features=6)
optimizer = optimizer = torch.optim.SGD(model.parameters(), lr=0)

checkpoint = torch.load(FILE)
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optim_state'])
epoch = checkpoint['epoch']

# Для использовавния
model.eval()
# Для обучения
model.train()
print(optimizer.state_dict())

# Вызвать model.eval(), чтобы установить dropout() и batch.normalization, иначе приведет к противоречивым результатам вывода.
# При возобновлении обучения, вызвать model.train().

""" 
Сохранение на GPU/CPU

1) Сохранение на GPU, загрузка на  CPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)
device = torch.device('cpu')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))

2) Сохранение на  GPU, Загрузка на GPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)
# Всегда использовать функцию .to(torch.device('cuda')) для всех входных данных модели.

3) Сохранить на CPU, Загрузить на GPU
torch.save(model.state_dict(), PATH)
device = torch.device("cuda")
model = Model(*args, **kwargs)
# "cuda:0" - выбрать нужный номер устройства с графическим процессором
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))
model.to(device)
# Вызвать model.to(torch.device('cuda')) для преобразования тензоров параметров модели в тензоры CUDA
"""