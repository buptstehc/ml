from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt

# 1.load sample data.
boston = datasets.load_boston()
X, y = boston.data, boston.target
m, n = np.shape(X)
y = y.reshape(m, 1)

# 2. normalize.
np.set_printoptions(precision=4)

u = np.mean(X, axis=0)
X = X - u

sigma = np.mean(X**2, axis=0)**.5
X = X / sigma

# add x0.
X = np.hstack((np.ones((m, 1)), X))

theta = (np.linalg.inv(X.T.dot(X))).dot(X.T).dot(y)
h = X.dot(theta)
J = np.sum((h-y)**2) / (2 * m)
print('the min mse is:%s\n' % J)