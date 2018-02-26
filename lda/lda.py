from sklearn import datasets
import numpy as np
from plot_data import plot_hist, plot_scatter

# 1.load and visualize data.
iris = datasets.load_iris()
X, y = iris.data, iris.target
#print np.shape(X), np.shape(y)
#plot_hist(X, y)

# 2.calculate Sw.
np.set_printoptions(precision=4)

mean_vectors = []
for cl in range(0,3):
    mean_vectors.append(np.mean(X[y==cl], axis=0))
    #print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl]))

S_W = np.zeros((4,4))
for i in range(0,3):
    # x,mean_vectors[i] are both row vectors here, reshape to column vectors.
    m = mean_vectors[i].reshape(4,1)

    for x in X[y==i]:
        x = x.reshape(4,1)
        S_W += (x - m).dot((x - m).T)

#print S_W

# 3.calculate Sb.
S_B = np.zeros((4,4))
sample_mean = np.mean(X, axis=0).reshape(4,1)
for i in range(0,3):
    m = mean_vectors[i].reshape(4,1)
    S_B += X[y==i].shape[0] * (m - sample_mean).dot((m - sample_mean).T)

#print S_B

# 4.calculate eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
#print eig_vals, eig_vecs
W = np.hstack((eig_vecs[:,0].reshape(4,1), eig_vecs[:,1].reshape(4,1)))
#print W

# 5.projection
X2 = X.dot(W)
plot_scatter(X2,y)