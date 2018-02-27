from sklearn import datasets
import numpy as np
from plot_data import plot_scatter


# 1.load sample data.
iris = datasets.load_iris()
X, y = iris.data, iris.target
m, n = np.shape(X)

# 2.normalize
u = np.mean(X, axis=0)
X = X - u

X_2 = X**2
sigma_2 = np.mean(X_2, axis=0)
sigma = sigma_2**.5
X = X / sigma

# 3.calculate 4*4 covariance matrix
C = X.T.dot(X) / m

# 4.calculate eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eig(C)
print eig_vals
U = np.hstack((eig_vecs[:,0].reshape(4,1), eig_vecs[:,1].reshape(4,1)))

# 5.projection
Y = X.dot(U)
plot_scatter(Y,y)