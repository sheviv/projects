import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# данные в 2d пространстве
X, _ = make_blobs(n_samples=10, centers=3, n_features=2,
                  cluster_std=0.2, random_state=0)

# создание индикатора для точек в данных
obj_names = []
for i in range(1, 11):
    obj = "Object " + str(i)
    obj_names.append(obj)
# создание DataFrame с именами и (x, y) координатами
data = pd.DataFrame({
    'Object': obj_names,
    'X_value': X[:, 0],
    'Y_value': X[:, -1]
})

# инициализация координат
c1 = (-1, 4)
c2 = (-0.2, 1.5)
c3 = (2, 2.5)


# функция для вычисления евклидова расстояния между точками данных и центроидами
def calculate_distance(centroid, X, Y):
    distances = []
    # распаковка координат x,y центроида
    c_x, c_y = centroid
    # обход точек данных и вычисление расстояния
    for x, y in list(zip(X, Y)):
        root_diff_x = (x - c_x) ** 2
        root_diff_y = (y - c_y) ** 2
        distance = np.sqrt(root_diff_x + root_diff_y)
        distances.append(distance)
    return distances


# рассчет расстояний и запись в DataFrame
data['C1_Distance'] = calculate_distance(c1, data.X_value, data.Y_value)
data['C2_Distance'] = calculate_distance(c2, data.X_value, data.Y_value)
data['C3_Distance'] = calculate_distance(c3, data.X_value, data.Y_value)
# выбор минимальных расстояний
data['Cluster'] = data[['C1_Distance', 'C2_Distance', 'C3_Distance']].apply(np.argmin, axis=1)
# сопоставление центроидов и переименование
data['Cluster'] = data['Cluster'].map({'C1_Distance': 'C1', 'C2_Distance': 'C2', 'C3_Distance': 'C3'})

# вычисление координат нового центроида из кластера 1
x_new_centroid1 = data[data['Cluster'] == 'C1']['X_value'].mean()
y_new_centroid1 = data[data['Cluster'] == 'C1']['Y_value'].mean()
# вычисление координат нового центроида из кластера 2
x_new_centroid2 = data[data['Cluster'] == 'C3']['X_value'].mean()
y_new_centroid2 = data[data['Cluster'] == 'C3']['Y_value'].mean()

# n_clusters - кол-во кластеров
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
# центроиды кластеров
print(kmeans.cluster_centers_)
# подписи кластеров
print(kmeans.labels_)
# нанесение центров кластеров и точек данных
plt.scatter(X[:, 0], X[:, -1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')
plt.title('Data points and cluster centroids')
plt.show()
