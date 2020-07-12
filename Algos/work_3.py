import numpy as np
import re
from matplotlib import pyplot as plt

data_dir = 'D:/UMass/Spring20/514/Assignments/'


def read_data(file):
    data = []
    labels = []
    f = open(file, 'r')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        label = re.findall(r'[A-Za-z]+.?[A-Za-z]+', line)
        row = re.findall(r'[0-9]+.*', line)
        labels.append(label[0])
        data.append([float(x) for x in row[0].strip().split(" ")])
    data = np.square(np.array(data))
    return data, labels


def recover_points(D, L):
    d = 2
    n = D.shape[0]
    PP_T = np.zeros_like(D)
    for i in range(PP_T.shape[0]):
        for j in range(PP_T.shape[1]):
            PP_T[i][j] = -0.5 * (D[i][j] - (1/n) * np.sum(D, axis=1)[i] - (1/n) * np.sum(D, axis=1)[j] + (1/n ** 2) * np.sum(D))

    eigenvalues, eigenvectors = np.linalg.eig(PP_T)
    P = np.matmul(eigenvectors[:, :d], np.sqrt(np.diag(eigenvalues[:d])))

    plt.figure()
    plt.title('Plot of cities')
    plt.scatter(P[:, 0], P[:, 1], marker=".", c="b")
    for i in range(len(P)):
        plt.annotate(L[i], (P[i][0], P[i][1]))
    plt.savefig(data_dir + 'p3_fig1')
    plt.show()

    # Inverting dim=1
    P[:, 1] = -1 * P[:, 1]
    plt.figure()
    plt.title('Plot of cities')
    plt.scatter(P[:, 0], P[:, 1], marker=".", c="r")
    for i in range(len(P)):
        plt.annotate(L[i], (P[i][0], P[i][1]))
    plt.savefig(data_dir + 'p3_fig2')
    plt.show()

    eigenvalues1, eigenvectors1 = np.linalg.eig(D)
    plt.plot(np.arange(1, n+1), np.abs(eigenvalues1))
    plt.title('Spectrum of D')
    plt.xticks(range(0, 21))
    plt.xlabel('Eigenvalue Rank')
    plt.ylabel('Eigenvalue')
    plt.savefig(data_dir + 'p3_fig3')
    plt.show()

    return True


def main():
    D, L = read_data(data_dir + 'UScities.txt')
    recover_points(D, L)
    exit(0)
    return


if __name__ == "__main__":
    main()