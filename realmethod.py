import matplotlib.pyplot as plt
import time
import pandas as pd

start_time = time.time()
data = pd.read_csv('data.csv')


def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        total_error += (y - (m * x + b)) ** 2

    total_error /= len(points)

    return total_error


def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b


m = 0
b = 0
Learning_Rate = 0.01
epochs = 1000

for i in range(epochs):
    m, b = gradient_descent(m, b, data, Learning_Rate)
    if i % 50 == 0:
        print(f"Epoch: {i}, slope: {m}, intercept: {b}")
print(f"Epoch: {epochs}, slope: {m}, intercept: {b}")
print(f"Loss Function: {loss_function(m, b, data)}")
print(f"Time taken: {time.time() - start_time}s")

plt.scatter(data.studytime, data.score, color="black")
plt.plot(range(0, 10), [m * x + b for x in range(0, 10)], color="red")
plt.show()
