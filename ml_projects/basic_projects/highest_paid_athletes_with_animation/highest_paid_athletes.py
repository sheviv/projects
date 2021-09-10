import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Стили визуализации
sns.set(style="darkgrid")
plt.style.use("seaborn-pastel")

# Если ошибка - установить openpyxl
df = pd.read_excel("data/Forbes Athlete List 2012-2019.xlsx", engine='openpyxl')
# print(df.head())

# Отчистить данные от символа #, $ и М.
df.Rank = df.Rank.apply(lambda x: int(x.split("#")[1]) if type(x) == np.str else x)
df.Pay = df.Pay.apply(lambda x: float(x.split(" ")[0].split("$")[1]))
df.Endorsements = df.Endorsements.apply(lambda x: float(x.split(" ")[0].split("$")[1]))
df["Salary/Winnings"].replace("-", '$nan M', inplace=True)
df["Salary/Winnings"] = df["Salary/Winnings"].apply(lambda x: float(x.split(" ")[0].split("$")[1]))
# Изменить имена переменных
df.Sport.replace({"Soccer": "Football",
                  "Football": "American Football",
                  "Mixed Martial Arts": "MMA",
                  "Auto racing": "Racing",
                  "Auto Racing": "Racing",
                  "Basketbal": "Basketball",
                  }, inplace=True)
df.columns = ['Rank', 'Name', 'Pay', 'Salary_Winnings', 'Endorsements', 'Sport', 'Year']


# Визуализация данных по видам спорта
# df.groupby("Name").first()["Sport"].value_counts().plot(kind="pie",autopct="%.0f%%",figsize=(8,8),wedgeprops=dict(width=0.4),pctdistance=0.8)
# plt.ylabel(None)
# plt.title("Breakdown of Athletes by Sport",fontweight="bold")
# plt.show()

"""
Анимация данных заработка
"""
# Преобразования столбца Year в объект DateTime
df.Year = pd.to_datetime(df.Year, format="%Y")
# Создание таблицы с индексами(ось x) - Year и строки(ось y) - Name.
racing_bar_data = df.pivot_table(values="Pay", index="Year", columns="Name")
# Поиск всех спортсменов, кто ежегодно в списке(без NaN)
# print(f"every_year: {racing_bar_data.columns[racing_bar_data.isnull().sum() == 0]}")
racing_bar_data.columns[racing_bar_data.isnull().sum() == 0]
# Преобразовать данные в сумму ЗП за все время
racing_bar_filled = racing_bar_data.interpolate(method="linear").fillna(method="bfill")
racing_bar_filled = racing_bar_filled.cumsum()

# Создание плавной(линейной интерполяцией) анимации
racing_bar_filled = racing_bar_filled.resample("1D").interpolate(method="linear")[::7]


"""
Создание анимации и сохранение(.gif)
"""

selected = racing_bar_filled.iloc[-1, :].sort_values(ascending=False)[:20].index
data = racing_bar_filled[selected].round()

fig, ax = plt.subplots(figsize=(9.3, 7))
fig.subplots_adjust(left=0.18)
no_of_frames = data.shape[0]

# Создание гистограммы первых строк данных
bars = sns.barplot(y=data.columns, x=data.iloc[0, :], orient="h", ax=ax)
ax.set_xlim(0, 1500)
txts = [ax.text(0, i, 0, va="center") for i in range(data.shape[1])]
title_txt = ax.text(650, -1, "Date: ", fontsize=12)
ax.set_xlabel("Pay (Millions USD)")
ax.set_ylabel(None)


def animate(i):
    y = data.iloc[i, :]
    # Изменение заголовков
    title_txt.set_text(f"Date: {str(data.index[i].date())}")
    # Обновление данных на графиках
    for j, b, in enumerate(bars.patches):
        # Обновление данных(ув.длины полосы)
        b.set_width(y[j])
        # Изменение текста данных(заработной платы) при ув.длины полосы
        txts[j].set_text(f"${y[j].astype(int)}M")
        txts[j].set_x(y[j])


anim = FuncAnimation(fig, animate, repeat=False, frames=no_of_frames, interval=1, blit=False)
anim.save('highest_paid_athletes.gif', writer='imagemagick', fps=120)
plt.close(fig)
