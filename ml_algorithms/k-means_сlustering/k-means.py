import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

# Get the Data
data = pd.read_csv("data.csv")

# data.head()

# data.info()

# pd.set_option('display.float', '{:.2f}'.format)
# data.describe()

"""
Exploratory Data Analysis
"""
# sns.scatterplot('room_board', 'grad_rate', data=data, hue='private')

# sns.scatterplot('outstate', 'f_undergrad', data=data, hue='private')

# plt.figure(figsize=(12, 8))
# data.loc[data.private == 'Yes', 'outstate'].hist(label="Private College", bins=30)
# data.loc[data.private == 'No', 'outstate'].hist(label="Non Private College", bins=30)
# plt.xlabel('Outstate')
# plt.legend()

# Гистограмма для столбца Grad.Rate
# plt.figure(figsize=(12, 8))
# data.loc[data.private == 'Yes', 'grad_rate'].hist(label="Private College", bins=30)
# data.loc[data.private == 'No', 'grad_rate'].hist(label="Non Private College", bins=30)
# plt.xlabel('Graduation Rate')
# plt.legend()
# plt.show()

"""
K Means Cluster Creation
"""
from sklearn.cluster import KMeans
# Создайте экземпляр модели KMeans с 2 кластерами.
# kmeans = KMeans(2)

# Обучение модели на всех данным, кроме Private.
# kmeans.fit(data.drop('private', axis=1))

# Векторы центров кластеров
# print(kmeans.cluster_centers_)

"""
Evaluation
"""
# Оценка кластеризации
data['private'] = data.private.astype("category").cat.codes
private = data.private
# print(kmeans.labels_)

# Отчет о классификации
from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(data.private, kmeans.labels_))
# print(classification_report(data.private, kmeans.labels_))

from sklearn.metrics import accuracy_score, confusion_matrix
# print(accuracy_score(data.private, kmeans.labels_))
# print(pd.DataFrame(classification_report(data.private, kmeans.labels_, output_dict=True)))

"""
Scaling the data
"""
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

X = data.drop('private', axis=1)
y = data.private
X = scalar.fit_transform(X)
kmeans = KMeans(2)
kmeans.fit(X)
print(kmeans.cluster_centers_)
print(accuracy_score(y, kmeans.labels_))
print(confusion_matrix(y, kmeans.labels_))
print(classification_report(y, kmeans.labels_))

