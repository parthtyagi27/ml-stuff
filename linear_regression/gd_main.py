import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import random

main_dataframe = pd.read_csv("placement_data.csv")

dataframe, test_dataframe = train_test_split(main_dataframe, test_size=0.2)
print(dataframe.describe())

theta_0 = random.random()
theta_1 = random.random()

learning_rate = 0.0004

print("Init theta 0 = " + str(theta_0))
print("Init theta 1 = " + str(theta_1))

def cost_function():
    hs_p = dataframe['hsc_p'].values
    hypothesis = (theta_0 * hs_p) + theta_1
    degree_p = dataframe['degree_p'].values
    d_0 = np.sum((hypothesis - degree_p) * hs_p)
    d_1 = np.sum((hypothesis - degree_p))
    return (d_0 / dataframe.shape[0], d_1 / dataframe.shape[0])

steps_per_epoch = 1000
epochs = 100

for i in range(epochs):
    for j in range(steps_per_epoch):
        d_theta_0, d_theta_1 = cost_function()
        theta_0 = theta_0 - (learning_rate * d_theta_0)
        theta_1 = theta_1 - (learning_rate * d_theta_1)
    print("Epoch {} of {}: theta_0 = {}, theta_1 = {}".format(i, epochs, theta_0, theta_1))
    
print(theta_0)
print(theta_1)

x_bar = dataframe.describe()['hsc_p']['mean']
y_bar = dataframe.describe()['degree_p']['mean']
x_min = dataframe.describe()['hsc_p']['min']
x_max = dataframe.describe()['hsc_p']['max']

x = np.linspace(x_min, x_max)
y = theta_0 * x + theta_1

plt.scatter(dataframe['hsc_p'], dataframe['degree_p'])
plt.plot(x, y, color="red")
plt.show()

# error = 0

# for index, row in test_dataframe.iterrows():
#     predict = b_1 * row['hsc_p'] + b_0
#     error = error + ((predict - row['degree_p'])**2)

# error = error / len(test_dataframe)
# error = math.sqrt(error)

# print("Error = " + str(error))