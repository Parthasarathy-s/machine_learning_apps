import numpy as np

x = np.array([5, 1, 0])           # features
w = np.array([0.4, 1.5, -2.0])    # weights
b = -1.0                          # bias

z = np.dot(x, w) + b              # 2.5
p = 1 / (1 + np.exp(-z))          # 0.924
print(z, p)
