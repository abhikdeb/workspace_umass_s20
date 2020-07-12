import pandas as pd
import autograd.numpy as np
from autograd import elementwise_grad
from matplotlib import pyplot as plt
import numpy

data_dir = "D:/UMass/Spring20/688/Homework/Assignment5/Assignment5/Data/"
glob = {}


def fetch_inputs():
    glob['X_train'] = np.array(pd.read_csv(data_dir + "X_train.csv", sep=" ", header=None))
    glob['Y_train'] = np.array(pd.read_csv(data_dir + "Y_train.csv", sep=" ", header=None))
    glob['X_test'] = np.array(pd.read_csv(data_dir + "X_test.csv", sep=" ", header=None))
    glob['Y_test'] = np.array(pd.read_csv(data_dir + "Y_test.csv", sep=" ", header=None))
    return True


def sigmoid(s):
    return 1 / (1 + np.exp(-1 * s))


def lae(s, t):
    return np.logaddexp(s, t)


def likelihood(y, x, z):
    return -1 * np.sum(lae(np.zeros((len(y), 1)), -1 * y * np.dot(x, z)))


def prior(z):
    return -0.5 * np.linalg.norm(z)**2 - np.log(np.sqrt(2 * np.pi))


def f_x(w):
    x = glob['X_train']
    y = glob['Y_train']
    eta = np.random.randn(5, 1)
    z = np.sqrt(0.5) * eta + w
    glob['z'] = z
    return prior(z) + likelihood(y, x, z)


def SGVI(w_init, t_max, step):
    w_hist = []
    z_hist = []
    d_fx = elementwise_grad(f_x)
    w = w_init.copy()
    w_hist.append(w.copy())
    for t in range(t_max):
        grad_f = d_fx(w)
        w += step * grad_f
        w_hist.append(w.copy())
        z_hist.append((glob['z']))
    return np.array(w_hist[1:]), np.array(z_hist)


def prob_4():
    x_train = glob['X_train']
    y_train = glob['Y_train']
    print('X_train : ', x_train.shape)
    print('Y_train : ', y_train.shape)
    hist_w, hist_z = SGVI(np.zeros((x_train.shape[1], 1)), 10000, 0.005)
    line1 = hist_w[:, 0, :]
    line2 = hist_w[:, 1, :]
    line3 = hist_w[:, 2, :]
    line4 = hist_w[:, 3, :]
    line5 = hist_w[:, 4, :]
    plt.plot(np.arange(0, hist_w.shape[0]), line1, label="w[0]")
    plt.plot(np.arange(0, hist_w.shape[0]), line2, label="w[1]")
    plt.plot(np.arange(0, hist_w.shape[0]), line3, label="w[2]")
    plt.plot(np.arange(0, hist_w.shape[0]), line4, label="w[3]")
    plt.plot(np.arange(0, hist_w.shape[0]), line5, label="w[4]")
    plt.legend()
    plt.title("w vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Value of w")
    plt.savefig('q4.png')
    plt.show()
    return True


def predict_score(y, x, z):
    y = np.squeeze(y)
    y_prob = np.mean(sigmoid(np.dot(x, z)), axis=1)
    y_pred = numpy.array(y_prob)
    y_pred[y_pred <= 0.5] = -1.0
    y_pred[y_pred > 0.5] = 1.0
    score = numpy.sum([y == y_pred]) / y.shape[0]
    return 1.0 - score


def prob_5():
    t_list = [10, 100, 1000, 10000]
    n_trials = 5
    x_train = glob['X_train']
    y_train = glob['Y_train']
    x_test = glob['X_test']
    y_test = glob['Y_test']

    for t in t_list:
        for n in range(n_trials):
            hist_w, hist_z = SGVI(np.zeros((x_train.shape[1], 1)), t, 0.005)
            w = hist_w[-1]
            eta = np.random.randn(5, 1000)
            z = np.sqrt(0.5) * eta + w
            # err_rate = predict_score(y_test, x_test, np.squeeze(hist_z).T)
            err_rate = predict_score(y_test, x_test, z)
            print("Iter : ", t, " | Trial : ", n, " | Error : ", err_rate)

    return True


def main():
    print("Executing...")
    fetch_inputs()
    prob_4()
    prob_5()
    exit(0)
    return True


if __name__ == "__main__":
    main()
