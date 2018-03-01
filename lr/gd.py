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

# 3. batch gradient descent
num_iters = 100;
alpha = 0.1
theta = np.zeros((n+1, 1))
J = np.zeros((num_iters, 1))

for i in range(0, num_iters):
    grad = (X.T).dot(X.dot(theta)-y)
    theta = theta - alpha * grad / m

    h = X.dot(theta)
    J[i] = np.sum((h-y)**2) / (2 * m)

# 4. plot the convergence graph.
plt.plot(np.arange(1, num_iters + 1), J)
plt.xlabel('number of iterations')
plt.ylabel('mean squared error')
plt.title('linear regression with gradient descent')
plt.show()