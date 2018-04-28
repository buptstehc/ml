import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. load and visualize data.
df_X = pd.read_fwf('logistic_x.txt', header=None, names=['x1', 'x2'])
df_y = pd.read_fwf('logistic_y.txt', header=None, names=['y'])
df_X = pd.concat([df_X, df_y], axis=1)

ax = plt.axes()
df_X.query('y == -1').plot.scatter(x=0, y=1, ax=ax, color='blue')
df_X.query('y == 1').plot.scatter(x=0, y=1, ax=ax, color='red')
# plt.show()

df_X = df_X.iloc[:,0:2] # recover

# 2. Newton-Raphson method.
X = df_X.values
m, n = X.shape
X = np.hstack((np.ones((m, 1)), X))
y = df_y.values

all_thetas = [] # collect for demonstration purpose
theta = np.zeros((n+1, 1))
tol = 1e9
n_iters = 0
while tol > 1e-10:
    Z = np.exp(X.dot(theta)*y)
    C = Z / np.power((1 + Z), 2)
    X2 = np.sqrt(C)*X
    H = (X2.T).dot(X2) / m

    grad = np.sum((-y / (1 + Z))*X, axis=0).reshape(n+1, 1) / m

    old_theta = theta.copy()
    theta = theta - (np.linalg.inv(H)).dot(grad)
    all_thetas.append(theta.copy())
    n_iters += 1
    tol = np.sum(np.abs(theta - old_theta))

print('converged after {0} iterations'.format(n_iters))

# 3. show the convergence process.
_xs = np.array([np.min(X[:,1]), np.max(X[:,1])])
for k, theta in enumerate(all_thetas):
    _ys = (theta[0] + theta[1] * _xs) / (- theta[2])
    plt.plot(_xs, _ys, lw=0.5)
plt.show()