import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

main_dataframe = pd.read_csv("placement_data.csv")

dataframe, test_dataframe = train_test_split(main_dataframe, test_size=0.2)

print(dataframe.describe())

x_bar = dataframe.describe()['hsc_p']['mean']
y_bar = dataframe.describe()['degree_p']['mean']
x_min = dataframe.describe()['hsc_p']['min']
x_max = dataframe.describe()['hsc_p']['max']

b_0 = 0
b_1_top = 0
b_1_bottom = 0

for index, row in dataframe.iterrows():
    x_i = row['hsc_p']
    y_i = row['degree_p']
    b_1_top = b_1_top + ((x_i - x_bar) * (y_i - y_bar))
    b_1_bottom = b_1_bottom + ((x_i - x_bar) ** 2)

b_1 = b_1_top / b_1_bottom
b_0 = y_bar - (b_1 * x_bar)

x = np.linspace(x_min, x_max)
y = b_1 * x + b_0

error = 0

for index, row in test_dataframe.iterrows():
    predict = b_1 * row['hsc_p'] + b_0
    error = error + ((predict - row['degree_p'])**2)

error = error / len(test_dataframe)
error = math.sqrt(error)

print("Error = " + str(error))


plt.scatter(dataframe['hsc_p'], dataframe['degree_p'])
plt.plot(x, y, color="red")
plt.show()