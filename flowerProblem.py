import numpy as np
from matplotlib import pyplot as plt

# Each data point is length, width, type
data = [[3, 1.5, 1],
        [2, 1, 0],
        [4, 1.5, 1],
        [3, 1, 0],
        [3.5, 0.5, 1],
        [2, 0.5, 0],
        [5.5, 1, 1],
        [1, 1, 0]]

mysteryFlower = [4.5, 1]

# network architecture
      #   o         flower type
      #  / \        w1, w2, b
      # o   o       m1, m2

w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_p(x):
    return sigmoid(x) * (1 - sigmoid(x))

T = np.linspace(-5, 5, 100)
Y = sigmoid(T)

# plt.plot(T, Y, c = 'r')
# plt.plot(T, sigmoid_p(T), c = 'b')
# plt.show()

# plot data
for j in range(len(data)):
    point = data[j]
    color = 'r'
    if point[2] == 0:
        color = 'b'
        pass
    plt.scatter(point[0], point[1], c=color)
    pass
# plt.show()

learningRate = 0.1

# Training loop
for i in range(10000):
    rndIdx = np.random.randint(len(data))
    point = data[rndIdx]

    z = point[0] * w1 + point[1] * w2 + b
    pred = sigmoid(z)

    target = point[2]
    cost = np.square(pred - target)

    dCost_dpred = 2 * (pred - target)
    dpred_dz = sigmoid_p(z)

    dz_dw1 = point[0]
    dz_dw2 = point[1]
    dz_db  = 1

    dcost_dz = dCost_dpred * dpred_dz

    dcost_dw1 = dcost_dz * dz_dw1
    dcost_dw2 = dcost_dz * dz_dw2
    dcost_db  = dcost_dz * dz_db

    w1 = w1 - learningRate * dcost_dw1
    w2 = w2 - learningRate * dcost_dw2
    b = b - learningRate * dcost_db
    pass

print(sigmoid(mysteryFlower[0] * w1 + mysteryFlower[1] * w2 + b))
