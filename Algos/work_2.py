from scipy.io import loadmat
# from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from sklearn import preprocessing
import numpy as np


def sim_hash(x, r):
    v = [0] * r
    for i in len(x):
        hsh = hash(x[i])
        
    return


def main():
    x = loadmat('D:/UMass/Spring20/514/Assignments/mnist.mat')
    print(type(x))
    print(x.keys())
    print(x['trainX'].shape)

    x_train = x['trainX']
    y_train = x['trainY']
    x_test = x['testX']
    y_test = x['testY']
    # Normalize to unit l2 norm
    x_train = preprocessing.normalize(x_train)

    print(x_train[0])
    print(np.linalg.norm(x_train[0], ord=2))

    hash_list = []
    for i in range(x_train.shape[0]):
        if cosine(x_train[0], x_train[i]) >= 0.95:
            # counter += 1a
            pass

    print()

    return


if __name__ == "__main__":
    main()
