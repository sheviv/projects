import torch
import jovian
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

# Проверка данных
DATA_FILENAME = "data/car_data.csv"
dataframe_raw = pd.read_csv(DATA_FILENAME)

# Слово для использования в качестве случайной строки
your_name = "Gagarin"  # не менее 5 символов
# Гиперпараметры
epochs = 90
lr = 1e-8


# Настройка и очистка данных(отсортировать строки и удалить столбцы, которые не нужны - названия автомобилей)
def customize_dataset(dataframe_raw, rand_str):
    dataframe = dataframe_raw.copy(deep=True)
    # выбросить несколько строк
    dataframe = dataframe.sample(int(0.95*len(dataframe)), random_state=int(ord(rand_str[0])))
    # масштаб ввода
    dataframe.Year = dataframe.Year * ord(rand_str[1])/100.
    # масштабировать целевое значение
    dataframe.Selling_Price = dataframe.Selling_Price * ord(rand_str[2])/100.
    # выбросить столбцы с именем
    if ord(rand_str[3]) % 2 == 1:
        dataframe = dataframe.drop(['Car_Name'], axis=1)
    return dataframe

dataframe = customize_dataset(dataframe_raw, your_name)

# print(dataframe.head())

input_cols = ["Year", "Present_Price", "Kms_Driven", "Owner"]
categorical_cols = ["Fuel_Type", "Seller_Type", "Transmission"]
output_cols = ["Selling_Price"]

"""
Обработка данных
"""
# Преобразовать данные из фрейма в тензоры PyTorch
# Преобразование в массивы NumPy:
def dataframe_to_arrays(dataframe):
    # Копия исходног фрейма
    dataframe1 = dataframe.copy(deep=True)
    # Преобразование нечисловых категориальных столбцов в числа
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    # Преобразовать входные и выходные данные в массивов numpy
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array

inputs_array, targets_array = dataframe_to_arrays(dataframe)

# Данные -> массивы NumPy -> тензоры PyTorch
inputs = torch.Tensor(inputs_array)
targets = torch.Tensor(targets_array)

dataset = TensorDataset(inputs, targets)
train_ds, val_ds = random_split(dataset, [228, 57])
batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

"""
Создание модели PyTorch
"""
# Модель линейной регрессии
input_size = len(input_cols)
output_size = len(output_cols)


class CarsModel(nn.Module):
    def __init__(self):
        super().__init__()
        # (input_size & output_size)
        # nn.Linear - линейная регрессия
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, xb):
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        inputs, targets = batch
        # Прогнозирование
        out = self(inputs)
        # Рассчитать фун-ию потерь
        # F.l1_loss - прогнозы и потери(параметр веса, равный смещению)
        loss = F.l1_loss(out, targets)
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        # Прогнозирование
        out = self(inputs)
        # Рассчитать фун-ию потерь
        # F.l1_loss - прогнозы и потери(параметр веса, равный смещению)
        loss = F.l1_loss(out, targets)
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        # Объединение фун-ии потерь(среднее ошибок)
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result, num_epochs):
        # Вывод инфо каждые n эпох
        if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch + 1, result['val_loss']))

model = CarsModel()
list(model.parameters())


"""
Обучающая модель
"""
# Оценка точности модели, после тренировки посмотреть, насколько уменьшаются потери.
# Алгоритм оценки
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# Алгоритм подгонки
def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Обучение модели
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Валидационные данные
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)
    return history


# Обучени модели с гиперапараметрами
history1 = fit(epochs, lr, model, train_loader, val_loader)

"""
Использование модели для прогнозирования
"""
# Фун-ия предсказания
def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)
    prediction = predictions[0].detach()
    print("Input:", input)
    print("Target:", target)
    print("Prediction:", prediction)

# Тестирование модели на некоторых образцах
input, target = val_ds[0]
predict_single(input, target, model)
input, target = val_ds[10]
predict_single(input, target, model)
