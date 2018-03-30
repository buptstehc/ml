from nltk.corpus import stopwords
import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import pickle

# remove punctuation and stopwords.
def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    return clean_words

def load_samples():
    '''
    1. use whole samples as dictionary
    2. return feature matrix X and labels y
    '''

    try:
        f = open('spam.m', 'r')
        return pickle.load(f)
    except IOError:
        pass

    y = []
    samples = []
    d = set([])
    f = open('spam.txt', "r")
    for line in f:
        text = process_text(line)

        samples.append(set(text[1:]))
        if text[0] == 'spam':
            y.append(1)
        else:
            y.append(0)

        d |= set(text[1:])

    d = list(d)
    m = len(y)
    n = len(d)
    X = np.zeros((m, n))
    for i in range(0, m):
        sample = samples[i]
        for j in range(0, n):
            if d[j] in sample:
                X[i][j] = 1

    # persistent samples here since initialize process cost too much.
    f = open('spam.m', 'w')
    pickle.dump((X, y), f)

    return (X, y)

# 1. load samples.
X, y = load_samples()

# choose 70% of X as train data, others as test data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
m, n = X_train.shape
y_train = np.array(y_train).reshape(m, 1)

# 2. parameter estimate using Laplace smoothing.
phi_y = sum(y_train) / float(m)
phi_1 = (sum(X_train * y_train) + 1) / (float(sum(y_train)) + 2)
phi_0 = (sum(X_train * (y_train ^ 1)) + 1) / ((m - float(sum(y_train))) + 2)

# 3. test
m, n = X_test.shape
y_test = np.array(y_test).reshape(m, 1)

p_1 = (X_test * phi_1 + (1 - phi_1) * (1 - X_test)).prod(axis = 1) * phi_y
p_0 = (X_test * phi_0 + (1 - phi_0) * (1 - X_test)).prod(axis = 1) * (1 - phi_y)
h = (p_1 >= p_0).reshape(m, 1)
print('accuracy rate: %.2f%%' % (np.sum(h == y_test) * 100 / float(m)))

# 4. compare with scikit
bnb = BernoulliNB()
h = bnb.fit(X_train, y_train.ravel()).predict(X_test)
print('accuracy rate of scikit: %.2f%%' % (bnb.fit(X_train, y_train.ravel()).score(X_test, y_test.ravel()) * 100))