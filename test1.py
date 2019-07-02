import numpy

def NN(m1, m2, w1, w2, b):
    z = m1 * w1 + m2 * w2 +b
    return sigmoid(z)

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def cost(b):
    return (b - 4) ** 2

w1 = numpy.random.randn()
w2 = numpy.random.randn()
b = numpy.random.randn()

# print(NN(3, 1.5, w1, w2, b))

def slope(b):
    return 2 * (b - 4)
    pass

b = 0
for i in range(10):
    b = b - 0.1 * slope(b)
    print(b)
    pass
