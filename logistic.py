from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import numpy as np

cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target

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

theta = np.zeros((n+1, 1))
num_iters = 1000;
alpha = 1e-3

for i in range(0, num_iters):
    grad = X_train.T.dot(y_train - (1 + np.exp(-X_train.dot(theta)))**(-1))
    theta = theta + alpha * grad

# 4. test
m, n = X_test.shape
y_test = y_test.reshape(m, 1)
X_test = np.hstack((np.ones((m, 1)), X_test)) # add x0.

h = (1 + np.exp(-X_test.dot(theta)))**(-1)
print('accuracy rate1: %.2f%%' % (np.sum(np.argmax(h, axis=1).reshape(m, 1) == y_test) * 100 / float(m)))

logistic = linear_model.LogisticRegression()
print('accuracy rate2: %.2f%%' % (logistic.fit(X_train, y_train.ravel()).score(X_test, y_test.ravel()) * 100))