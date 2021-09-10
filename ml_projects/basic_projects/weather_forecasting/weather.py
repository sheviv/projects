import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

df = pd.read_csv("data/Weather.csv")
# print(df.head())
# Не использовать столбец Unnamed
df = pd.read_csv("data/Weather.csv", index_col=0)


# Задать атрибут, содержащий дату(для получения значения температуры с временной шкалой).
df1 = pd.melt(df, id_vars='YEAR', value_vars=df.columns[1:])

df1['Date'] = df1['variable'] + ' ' + df1['YEAR'].astype(str)
# Преобразование String в объект datetime
df1.loc[:, 'Date'] = df1['Date'].apply(lambda x: datetime.strptime(x, '%b %Y'))

"""
Температура по времени
"""
df1.columns = ['Year', 'Month', 'Temperature', 'Date']
# Чтобы получить правильный временной ряд.
df1.sort_values(by='Date', inplace=True)
# fig = go.Figure(layout=go.Layout(yaxis=dict(range=[0, df1['Temperature'].max() + 1])))
# fig.add_trace(go.Scatter(x=df1['Date'], y=df1['Temperature']), )
# fig.update_layout(title='Temperature Through Timeline:', xaxis_title='Time', yaxis_title='Temperature in Degrees')
# fig.update_layout(xaxis=go.layout.XAxis(rangeselector=dict(buttons=list([dict(label="Whole View", step="all"),
#                                                                          dict(count=1, label="One Year View", step="year", stepmode="todate")])),
#                                         rangeslider=dict(visible=True), type="date"))
# fig.show()

"""
Четыре основных сезона в Индии:
- Зима: декабрь, январь и февраль.
- Лето: март, апрель и май.
- Муссон: июнь, июль, август и сентябрь.
- Осень: октябрь и ноябрь.
"""
# Самая теплая, самая холодная и средняя месячная температура.
# fig = px.box(df1, 'Month', 'Temperature')
# fig.update_layout(title='Warmest, Coldest and Median Monthly Temperature.')
# fig.show()


"""
Визуализация данных в браузере
"""
# Кластеризация KMeans
# Оценка по количеству кластеров(Количество кластеров/ Сумма квадратов расстояния)
from sklearn.cluster import KMeans
# sse = []
# target = df1['Temperature'].to_numpy().reshape(-1,1)
# num_clusters = list(range(1, 10))
# for k in num_clusters:
#     km = KMeans(n_clusters=k)
#     km.fit(target)
#     sse.append(km.inertia_)
# fig = go.Figure(data=[
#     go.Scatter(x = num_clusters, y=sse, mode='lines'),
#     go.Scatter(x = num_clusters, y=sse, mode='markers')
# ])
# fig.update_layout(title="Evaluation on number of clusters:",
#                  xaxis_title = "Number of Clusters:",
#                  yaxis_title = "Sum of Squared Distance",
#                  showlegend=False)
# fig.show()

# По графику можно выбрать 3 или 4 размер кластеров
# km = KMeans(4)
km = KMeans(3)

# Визуализация температуры от даты
km.fit(df1['Temperature'].to_numpy().reshape(-1, 1))
df1.loc[:, 'Temp Labels'] = km.labels_
# fig = px.scatter(df1, 'Date', 'Temperature', color='Temp Labels')
# fig.update_layout(title = "Temperature clusters.",
#                  xaxis_title="Date", yaxis_title="Temperature")
# fig.show()


# Частота "появления"температуры(подсчет температуры в данных)
# fig = px.histogram(x=df1['Temperature'], nbins=200, histnorm='density')
# fig.update_layout(title='Frequency chart of temperature readings:',
#                  xaxis_title='Temperature', yaxis_title='Count')
# fig.show()
# Вывод: среднегодовая температура 26,8-26,9.

"""
Среднегодовая температура 
"""
# df['Yearly Mean'] = df.iloc[:,1:].mean(axis=1)
# fig = go.Figure(data=[
#     go.Scatter(name='Yearly Temperature' , x=df['YEAR'], y=df['Yearly Mean'], mode='lines'),
#     go.Scatter(name='Yearly м' , x=df['YEAR'], y=df['Yearly Mean'], mode='markers')
# ])
# fig.update_layout(title='Yearly Mean Temperature :',
#                  xaxis_title='Time', yaxis_title='Temperature in Degrees')
# fig.show()
# Вывод:
# - Среднегодовая температура не повышалась до 1980 года.
# - После 2015г. - температура сильно повысилась.
# - Тенденция к повышению-понижению годовой температуры.


"""
Месячные температуры в истории
"""
# fig = px.line(df1, 'Year', 'Temperature', facet_col='Month', facet_col_wrap=4)
# fig.update_layout(title='Monthly Temperature throughout history:')
# fig.show()
# Вывод: подтверждение тенденции ув. температуры

"""
Сезонный анализ погоды
"""
# Средние сезонные(Зима, Лето, Муссон, Осень) температуры в течение всего периода.
# df['Winter'] = df[['DEC', 'JAN', 'FEB']].mean(axis=1)
# df['Summer'] = df[['MAR', 'APR', 'MAY']].mean(axis=1)
# df['Monsoon'] = df[['JUN', 'JUL', 'AUG', 'SEP']].mean(axis=1)
# df['Autumn'] = df[['OCT', 'NOV']].mean(axis=1)
# seasonal_df = df[['YEAR', 'Winter', 'Summer', 'Monsoon', 'Autumn']]
# seasonal_df = pd.melt(seasonal_df, id_vars='YEAR', value_vars=seasonal_df.columns[1:])
# seasonal_df.columns=['Year', 'Season', 'Temperature']
# fig = px.scatter(seasonal_df, 'Year', 'Temperature', facet_col='Season', facet_col_wrap=2, trendline='ols')
# fig.update_layout(title='Seasonal mean temperatures throughout years:')
# fig.show()
# Вывод: тенденция ув.температуры за весь период в пределах 1-2 градусов.


"""
Прогнозирование погоды
"""
# Прогноз среднемесячной погоды на 2018г(1900-2017гг в данных)
# Прогнозирование через дерево решений(данные не линейны)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df2 = df1[['Year', 'Month', 'Temperature']].copy()
df2 = pd.get_dummies(df2)
y = df2[['Temperature']]
x = df2.drop(columns='Temperature')

dtr = DecisionTreeRegressor()
train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3)
dtr.fit(train_x, train_y)
pred = dtr.predict(test_x)
# Если r2 стремится к 1 - модель работает хорошо, иначе изменить.
r2_score(test_y, pred)


# Взять данные 2017г.
year_2018 = df1[df1['Year'] == 2017][['Year', 'Month']]
# Замена значений 2017 на 2018
year_2018.Year.replace(2017, 2018, inplace=True)
# Преобразовать категориальные значения(2018) в фиктивные.
year_2018 = pd.get_dummies(year_2018)
temp_2018 = dtr.predict(year_2018)

temp_2018 = {'Month': df1['Month'].unique(), 'Temperature': temp_2018}
temp_2018 = pd.DataFrame(temp_2018)
temp_2018['Year'] = 2018
# print(temp_2018)

"""
Среднегодовая температура 2018г
"""
# Визуализация предсказанных данных
forecasted_temp = pd.concat([df1, temp_2018], sort=False).groupby(by='Year')['Temperature'].mean().reset_index()
fig = go.Figure(data=[
    go.Scatter(name='Yearly Mean Temperature', x=forecasted_temp['Year'], y=forecasted_temp['Temperature'], mode='lines'),
    go.Scatter(name='Yearly Mean Temperature', x=forecasted_temp['Year'], y=forecasted_temp['Temperature'], mode='markers')
])
fig.update_layout(title='Forecasted Temperature:',
                  xaxis_title='Time', yaxis_title='Temperature in Degrees')
fig.show()
