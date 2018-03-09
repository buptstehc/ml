from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# 1. init.
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 2. normalize.
np.set_printoptions(precision=4)

u = np.mean(X, axis=0)
X = X - u

sigma = np.mean(X**2, axis=0)**.5
X = X / sigma

# choose 70% of X as train data, others as test data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 3. train
m, n = X_train.shape
y_train = y_train.reshape(m, 1)
X_train = np.hstack((np.ones((m, 1)), X_train)) # add x0.

k = 3 # classification size.
theta = np.zeros((n+1, k))
num_iters = 1000;
alpha = 1e-5

for i in range(0, num_iters):
    grad_2_1 = np.sum(np.exp(X_train.dot(theta)))

    for s in range(0, k):
        grad_1 = np.sum(X_train * (y_train == s), axis=0)
        grad_2_2 = np.sum(X_train * np.exp(X_train.dot(theta[:, s])).reshape(m, 1))
        theta[:, s] = theta[:, s] + alpha * (grad_1 - grad_2_2 / grad_2_1)

# 4. test
m, n = X_test.shape
y_test = y_test.reshape(m, 1)
X_test = np.hstack((np.ones((m, 1)), X_test)) # add x0.

h = np.exp(X_test.dot(theta)) / np.sum(np.exp(X_test.dot(theta)), axis=1).reshape(m, 1)
print('accuracy rate: %.2f%%' % (np.sum(np.argmax(h, axis=1).reshape(m, 1) == y_test) * 100 / float(m)))