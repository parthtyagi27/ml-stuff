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

"""
INTUITION:
Let h(x) = theta_0 + theta_1*x_1 + ... where x is a vector of features

Define cost function to be sum[]
"""

theta = np.ones((3, 1))
print(theta)
learning_rate = 0.0001

x_i = dataframe.iloc[:, [2, 4]].to_numpy()
ones = np.ones((x_i.shape[0], 1))
x_i = np.hstack((ones, x_i))

def gradient_descent():
    hypothesis = x_i.dot(theta)
    degree_p = dataframe['degree_p'].values
    diff = np.sum(hypothesis - degree_p)
    for i in range(theta.shape[0]):
        multiplier = 0
        if i == 1: 
            multiplier = 1
        else:
            multiplier = theta[i, 0]
        theta[i, 0] = theta[i, 0] - ((learning_rate / dataframe.shape[0]) * diff * multiplier)

epochs = 10
steps_per_epoch = 100

def getError():
    test_data = test_dataframe['degree_p'].values
    test_x_i = test_dataframe.iloc[:, [2, 4]].to_numpy()
    ones = np.ones((test_x_i.shape[0], 1))
    test_x_i = np.hstack((ones, test_x_i))
    return np.sum(test_x_i.dot(theta) - test_data)

for i in range(epochs):
    for j in range(steps_per_epoch):
        gradient_descent()
    print("Epoch {} of {}: theta_vec {}".format(i, epochs, theta))
    print("Error = {}".format(getError()))
