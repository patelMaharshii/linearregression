import math
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

data = pd.read_csv("data.csv")
start_time = time.time()
set_a = data["score"]
set_b = data["studytime"]


def mean(values):
    return sum(values)/len(values)


mean_a = mean(set_a)
mean_b = mean(set_b)


def standard_deviation(the_mean, values):
    the_sum = 0
    for val in values:
        the_sum += (val - the_mean)**2
    the_sum /= len(values)
    return math.sqrt(the_sum)


std_a = standard_deviation(mean_a, set_a)
std_b = standard_deviation(mean_b, set_b)


def correlation(x_mean, y_mean, std_x, std_y, set_x, set_y):
    the_sum = 0
    n = len(set_x)
    for i in range(n):
        x_part = (set_x[i] - x_mean)/std_x
        y_part = (set_y[i] - y_mean)/std_y
        the_sum += (x_part*y_part)

    return the_sum/(n-1)


r_value = correlation(mean_b, mean_a, std_b, std_a, set_b, set_a)
the_slope = r_value * (std_a / std_b)
the_intercept = mean_a - (the_slope * mean_b)


def f(x):
    return the_slope * x + the_intercept


def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        total_error += (y - (m * x + b)) ** 2

    total_error /= len(points)

    return total_error


print(f"Slope: {the_slope}, Intercept: {the_intercept}")
print(f"Loss Function: {loss_function(the_slope, the_intercept, data)}")
print(f"Time taken: {time.time() - start_time}s")
x = np.arange(0, 10)
plt.scatter(set_b, set_a)
plt.plot(x, f(x))
plt.show()
