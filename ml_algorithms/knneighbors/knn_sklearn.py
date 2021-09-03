import numpy as np
import pylab as pl
from sklearn import neighbors, datasets
# импорт набора данных
iris = datasets.load_iris()
X = iris.data[:, :2]  # мы берем только первые две особенности
Y = iris.target
h = .02  # размер шага в сетке
knn = neighbors.KNeighborsClassifier()
# создание экземпляра классификатора соседей и подгон данных
knn.fit(X, Y)
# Построение границ решения(назначение цвеа каждой точке сетки)
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
pl.figure(1, figsize=(4, 3))
pl.set_cmap(pl.cm.Paired)
pl.pcolormesh(xx, yy, Z)
# построение тренировочных точек
pl.scatter(X[:, 0], X[:, 1], c=Y)
pl.xlabel('Sepal length')
pl.ylabel('Sepal width')

pl.xlim(xx.min(), xx.max())
pl.ylim(yy.min(), yy.max())
pl.xticks(())
pl.yticks(())
pl.show()
