import torch
import numpy as np

# В pytorch основано на тензорных операциях.
# Тензор может иметь разные размеры(1d, 2d или даже 3d и выше)

# скаляр, вектор, матрица, тензор
# torch.empty(size): uninitiallized
x = torch.empty(1)  # scalar
print(x)
x = torch.empty(3)  # vector, 1D
print(x)
x = torch.empty(2, 3)  # matrix, 2D
print(x)
x = torch.empty(2, 2, 3)  # tensor, 3 size
# x = torch.empty(2,2,2,3) # tensor, 4 size
print(x)

# torch.rand(size): random numbers [0, 1]
x = torch.rand(5, 3)
print(x)

# torch.zeros(size), fill with 0
# torch.ones(size), fill with 1
x = torch.zeros(5, 3)
print(x)

# check size
print(x.size())

# check data type
print(x.dtype)

# specify types, float32 default
x = torch.zeros(5, 3, dtype=torch.float16)
print(x)

# check type
print(x.dtype)

# construct from data
x = torch.tensor([5.5, 3])
print(x.size())

# requires_grad argument
# Это скажет pytorch, что ему нужно будет вычислить градиенты для тензора(на шагах оптимизации)
# т.е. переменная в модели, которую можно оптимизировать
x = torch.tensor([5.5, 3], requires_grad=True)

# Operations
y = torch.rand(2, 2)
x = torch.rand(2, 2)

# element wise addition
z = x + y
# torch.add(x,y)

# все, что заканчивается подчеркиванием, является локальной операцией, т.е. он изменит переменную
# y.add_(x)

# subtraction
z = x - y
z = torch.sub(x, y)

# multiplication
z = x * y
z = torch.mul(x, y)

# division
z = x / y
z = torch.div(x, y)

# Slicing
x = torch.rand(5, 3)
print(x)
print(x[:, 0])  # all rows, column 0
print(x[1, :])  # row 1, all columns
print(x[1, 1])  # element at 1, 1

# Фактическое значение, если в тензоре только 1 элемент
print(x[1, 1].item())

# Reshape with torch.view()
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# если -1, pytorch автоматически определит необходимый размер
print(x.size(), y.size(), z.size())

# Numpy
# # Converting a Torch Tensor to a NumPy array and vice versa is very easy
a = torch.ones(5)
b = a.numpy()
print(b)
print(type(b))

# numpy to torch with .from_numpy(x)
a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

# все тензоры создаются на CPU, можно переместить их на GPU(если он доступен)
if torch.cuda.is_available():
    device = torch.device("cuda")  # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)  # or .to("cuda")
    z = x + y
    # z = z.numpy() # numpy не может обрабатывать теноры графического процессора
    # move to CPU again
    z.to("cpu")
    # z = z.numpy()

"""
autograd
"""
# autograd обеспечивает автоматическую дифференциацию для всех операций над тензорами.

# requires_grad = True -> отслеживать все операции с тензором.
x = torch.randn(3, requires_grad=True)
y = x + 2

# y - создан в результате операции, поэтому имеет атрибут grad_fn.
# grad_fn: ссылается на функцию, которая создала тензор
print(x)  # созданный пользователем -> grad_fn is None
print(y)
print(y.grad_fn)

z = y * y * 3
print(z)
z = z.mean()
print(z)

# Градиенты с обратным распространением
# Когда закончатся вычисления, можно вызвать .backward(), и все градиенты будут вычислены автоматически.
# Градиент для этого тензора будет накапливаться в атрибуте .grad
z.backward()
print(x.grad)  # dz/dx

# torch.autograd - это движок для вычисления векторно-якобианского произведения.
# Вычисляет частные производные при применении цепного правила

# -------------
# Модель с нескалярным выходом:
# Если тензор не скалярный(более 1 элемента), нужно указать аргументы для backward().
# Указать аргумент градиента, который является тензором соответствующей формы.
# Нужно для векторно-якобиевского произведения.
x = torch.randn(3, requires_grad=True)
y = x * 2
for _ in range(10):
    y = y * 2
print(y)
print(y.shape)
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float32)
y.backward(v)
print(x.grad)

# -------------
# Остановить тензор из истории отслеживания:
# Во время цикла обучения, когда обновляются веса(эта операция обновления не должна быть частью вычисления градиента)
# - x.requires_grad_(False)
# - x.detach()
# - заключить в 'с torch.no_grad():'

# .requires_grad_(...) changes an existing flag in-place.
a = torch.randn(2, 2)
print(a.requires_grad)
b = ((a * 3) / (a - 1))
print(b.grad_fn)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# .detach(): получить новый тензор с тем же содержимым, но без вычисления градиента:
a = torch.randn(2, 2, requires_grad=True)
print(a.requires_grad)
b = a.detach()
print(b.requires_grad)

# wrap in 'with torch.no_grad():'
a = torch.randn(2, 2, requires_grad=True)
print(a.requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)

# -------------
# backward() накапливает градиент для тензора в атрибуте .grad
# .zero_() - очистка градиентов перед новым шагом оптимизации
weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_output = (weights * 3).sum()
    model_output.backward()
    print(weights.grad)
    # оптимизировать модель, т.е. отрегулировать веса
    with torch.no_grad():
        weights -= 0.1 * weights.grad
    # влияет на окончательный вес и результат
    weights.grad.zero_()

print(weights)
print(model_output)

# Optimizer has zero_grad() method
# optimizer = torch.optim.SGD([weights], lr=0.1)
# During training:
# optimizer.step()
# optimizer.zero_grad()