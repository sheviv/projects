import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("data/news.csv")
# print(data.columns)
# print(data.head())

# x(заголовки новостей) - для обучения модели
x = np.array(data["title"])
# y(метка новостей)
y = np.array(data["label"])
# CountVectorizer() - преобразовывает коллекцию текстовых данных в матрицу токенов
cv = CountVectorizer()
x = cv.fit_transform(x)


# Разделение данных и обучение модели
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

# Предсказание подлинности заголовка
news_headline = "CA Exams 2021: Supreme Court asks ICAI to extend opt-out option for July exams, final order tomorrow"
data = cv.transform([news_headline]).toarray()
print(model.predict(data))
news_headline = "Cow dung can cure Corona Virus"
data = cv.transform([news_headline]).toarray()
print(model.predict(data))
