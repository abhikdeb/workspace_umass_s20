"""
    LINEAR CRF MODELLING - SEQUENCE LABELING
"""


import itertools as itr
import numpy as np
from scipy.optimize import minimize
import time
import matplotlib.pyplot as plt

labels = 'etainoshrd'
labels_dict = {labels.index(c): c for c in labels}
labels_pos = {c: labels.index(c) for c in labels}
home_dir = 'D:\\UMass\\Spring20\\688\\Homework\\Assignment3\\Assignment3/'
test_labels = np.loadtxt(home_dir + 'data/test_words.txt', dtype=str)
train_labels = np.loadtxt(home_dir + 'data/train_words.txt', dtype=str)

cache = None
X_train = None
X_test = None
test_flag = False

home_dir1 = 'D:\\UMass\\Spring20\\688\\Homework\\Assignment2\\Assignment2/'
feature_params = np.loadtxt(home_dir1+'model/feature-params.txt')
transition_params = np.loadtxt(home_dir1+'model/transition-params.txt')
feature_grad = np.loadtxt(home_dir1+'model/feature-gradient.txt')
transition_grad = np.loadtxt(home_dir1+'model/transition-gradient.txt')


def read_test(num):
    vec = np.loadtxt(home_dir + 'data/test_img' + str(num) + '.txt')
    return vec


def read_train(num):
    vec = np.loadtxt(home_dir + 'data/train_img' + str(num) + '.txt')
    return vec


def potential(x, wf, debug=False):
    L, F = x.shape
    potentials = []
    for i in range(L):
        pot = np.exp(wf.dot(x[i].T))
        potentials.append(pot)
        if debug:
            print(i, ' | ', labels_dict[pot.argmax()], ' | ', np.log(pot))
    return np.array(potentials)


def energy_func(y, x, wf, wt, debug=False):
    energy = 0.0
    L, F = x.shape
    y_true = list(y)
    pots = potential(x, wf)
    for j in range(L):
        idx = labels_pos[y_true[j]]
        idx_1 = labels_pos[y_true[j + 1]] if j < L - 1 else -1
        if debug:
            print(j, ' | ', y_true[j], ' | ', idx, ' | ', idx_1)
        if j == L - 1:
            energy += np.log(pots[j][idx])
        if j < L - 1:
            energy += np.log(pots[j][idx]) + wt[idx][idx_1]
    return energy


def partition(X, wf, wt, debug=False):
    parts = []
    for i in range(X.shape[0]):
        possible_combos = [''.join(list(map(''.join, p))) for p in itr.product(labels, repeat=X[i].shape[0])]
        part = 0.0
        for y in possible_combos:
            part += np.exp(energy_func(y, X[i], wf, wt))
        parts.append(part)
    if debug:
        print('Log Partition (Z) : ', np.log(parts))
    return np.array(parts)


def joint_probability(x, wf, wt, part):
    probs = []
    prob_word = None
    prob_max = -1.0
    possible_combos = [''.join(list(map(''.join, p))) for p in itr.product(labels, repeat=x.shape[0])]
    for s in possible_combos:
        prob = np.exp(energy_func(s, x, wf, wt)) / part
        probs.append(prob)
        if prob > prob_max:
            prob_max = prob
            prob_word = s
    return prob_word, prob_max, probs


def marginal_prob(joint_prob, y):
    possible_combos = [''.join(list(map(''.join, p))) for p in itr.product(labels, repeat=len(y))]
    dist = []
    for i in range(len(y)):
        probs = []
        for c in labels:
            prob = 0.0
            for j in range(len(joint_prob)):
                if possible_combos[j][i] == c:
                    prob += joint_prob[j]
                pass
            probs.append(prob)
            pass
        print(i, ' | ', np.array(probs))
        dist.append(probs)
        pass
    return dist


def message_passing(x, params):
    (wf, wt) = params
    L, F = x.shape
    x_pots = potential(x, wf)

    # forward pass
    forward_messages = []
    z_fwd = 0.0
    for i in range(L):
        if i == 0:
            forward_messages.append(x_pots[i].dot(np.exp(wt)))
        elif i == L - 1:
            forward_messages.append(x_pots[i] * forward_messages[i - 1])
            z_fwd = x_pots[i].dot(forward_messages[i - 1].T)
        else:
            forward_messages.append((x_pots[i] * forward_messages[i - 1]).dot(np.exp(wt)))

    forward_messages = np.array(forward_messages)

    # backward pass
    backward_messages = np.array([np.zeros(wt.shape[0])] * forward_messages.shape[0])
    z_bck = 0.0
    for i in range(L - 1, -1, -1):
        if i == L - 1:
            backward_messages[i, :] = x_pots[i].dot(np.exp(wt.T))
        elif i == 0:
            backward_messages[i, :] = x_pots[i] * backward_messages[i + 1]
            z_bck = x_pots[i].dot(backward_messages[i + 1].T)
        else:
            backward_messages[i, :] = (x_pots[i] * backward_messages[i + 1]).dot(np.exp(wt.T))
    backward_messages = np.array(backward_messages)

    return forward_messages[:-1], z_fwd, backward_messages[1:], z_bck, x_pots


def marginals(f_msg, b_msg, z, pots, params):
    (wf, wt) = params
    L = pots.shape[0]

    # marginal prob
    marginal_p = []
    for i in range(L):
        if i == 0:
            marginal_p.append(pots[i] * b_msg[i] / z)
        elif i == L - 1:
            marginal_p.append(f_msg[i - 1] * pots[i] / z)
        else:
            marginal_p.append(f_msg[i - 1] * pots[i] * b_msg[i] / z)

    # pairwise prob
    pairwise_p = []
    for i in range(L - 1):
        pm = np.zeros((10, 10))
        for j in range(10):
            for k in range(10):
                if i == 0:
                    pm[j][k] = b_msg[i + 1][k] * np.exp(wt)[j][k] * (1 / z) * pots[i][j] * pots[i + 1][k]
                elif i == L - 2:
                    pm[j][k] = f_msg[i - 1][j] * np.exp(wt)[j][k] * (1 / z) * pots[i][j] * pots[i + 1][k]
                else:
                    pm[j][k] = f_msg[i - 1][j] * b_msg[i + 1][k] * np.exp(wt)[j][k] * (1 / z) * pots[i][j] * \
                               pots[i + 1][k]
        pairwise_p.append(pm)

    return np.array(marginal_p), np.array(pairwise_p)


def f_x(w, N):
    global cache, X_train, test_flag
    log_lh = 0.0
    wf = np.reshape(w[:len(labels) * 321], (len(labels), 321))
    wt = np.reshape(w[len(labels) * 321:], (len(labels), len(labels)))
    cache = {}
    x_in = X_train[:N]
    if test_flag:
        x_in = X_test
    probs, probm = [], []
    params = (wf, wt)
    for i in range(len(x_in)):
        f_msg, fz, b_msg, bz, x_pot = message_passing(x_in[i], params)
        prob_s, prob_m = marginals(f_msg, b_msg, fz, x_pot, params)
        y_i = train_labels[i]
        if test_flag:
            y_i = test_labels[i]
        try:
            log_lh += np.log(prob_m[0][labels_pos[y_i[0]]][labels_pos[y_i[1]]])
        except:
            print(wf, wt)
        for j in range(1, len(x_in[i]) - 1):
            log_lh += np.log(prob_m[j][labels_pos[y_i[j]]][labels_pos[y_i[j + 1]]]) - np.log(
                prob_s[j][labels_pos[y_i[j]]])
        if not test_flag:
            probs.append(prob_s)
            probm.append(prob_m)

    if test_flag:
        log_lh /= len(x_in)
        return log_lh
    else:
        log_lh /= -1 * N
    cache = {'probs': np.array(probs), 'probm': np.array(probm), 'params': params}
    return log_lh


def df_x(w, N):
    global cache, X_train
    wf = np.reshape(w[:len(labels) * 321], (len(labels), 321))
    wt = np.reshape(w[len(labels) * 321:], (len(labels), len(labels)))
    x = X_train[:N]
    # wf, wt = cache['params']
    probs = cache['probs']
    probm = cache['probm']
    dWf = np.zeros_like(wf)
    dWt = np.zeros_like(wt)

    for i in range(N):
        x_i = x[i]
        y_i = train_labels[i]
        prob_s = probs[i]
        prob_m = probm[i]

        # dWf
        for m in range(dWf.shape[0]):
            for n in range(dWf.shape[1]):
                for l in range(len(x_i)):
                    dWf[m][n] += (y_i[l] == labels[m]) * x_i[l][n] - prob_s[l][m] * x_i[l][n]

        # dWt
        for m in range(dWt.shape[0]):
            for n in range(dWt.shape[1]):
                for l in range(len(x_i) - 1):
                    dWt[m][n] += ((y_i[l] == labels[m] and y_i[l + 1] == labels[n]) - prob_m[l][m][n])

    dWf /= -1 * N
    dWt /= -1 * N
    return np.append(dWf.flatten(), dWt.flatten())


def optimizer(n, debug=False):
    init_weights = np.zeros(len(labels) * 321 + len(labels) * len(labels))
    optimal_weights = minimize(f_x, init_weights, args=n, jac=df_x,
                               method='L-BFGS-B', options={'disp': debug, 'gtol': 1e-7}, tol=1e-7)
    print(optimal_weights)
    return optimal_weights.x


def get_score(weight):
    global test_labels, X_test
    predicted = []
    params = (weight[:3210].reshape((len(labels), 321)), weight[3210:].reshape((len(labels), len(labels))))
    for i in range(len(X_test)):
        f_msg, fz, b_msg, bz, xpot = message_passing(X_test[i], params)
        prob_s, prob_m = marginals(f_msg, b_msg, fz, xpot, params)
        predicted.append([labels_dict[idx] for idx in np.argmax(prob_s, axis=1)])
    corr = 0
    tot = 0
    for k in range(len(test_labels)):
        tot += len(test_labels[k])
        for l in range(len(test_labels[k])):
            if test_labels[k][l] == predicted[k][l]:
                corr += 1
    acc = (corr / tot)
    return acc


def prob_1(N_lst, plot_flag):
    print("Question 1.1.....")
    weights = []
    times = []
    for n in N_lst:
        start = time.time()
        opt_wt = optimizer(n, False)
        weights.append(opt_wt)
        np.save(home_dir+'weights_'+str(n)+'.txt', opt_wt)
        end = time.time()
        times.append(end - start)
        print(n, ' | time : ', (end - start), ' seconds')
    print(times)
    if plot_flag:
        plt.plot(N_lst, times, '--bo')
        plt.xlabel('Training set size')
        plt.ylabel('Training Time (in seconds)')
        plt.title('Time in seconds vs Training set size')
        plt.savefig(home_dir + 'fig1.png')
        plt.show()
    return weights


def prob_2(weights, N_lst, plot_flag):
    print("Question 1.2.....")
    scr_lst = []
    for i in range(len(N_lst)):
        scr = get_score(weights[i])
        scr_lst.append(1.0 - scr)
        print(N_lst[i], ' | ', 1.0 - scr)
    if plot_flag:
        plt.plot(N_lst, scr_lst, '--bo')
        plt.xlabel('Training set size')
        plt.ylabel('Prediction Error')
        plt.title('Prediction Error vs Training set size')
        plt.savefig(home_dir + 'fig2.png')
        plt.show()
    return True


def prob_3(weights, N, plot_flag):
    global test_flag
    print("Question 1.3.....")
    logs = []
    test_flag = True
    for i in range(len(N)):
        lh = f_x(weights[i], N[i])
        logs.append(lh)
        print(N[i], ' | ', lh)
    if plot_flag:
        plt.plot(N, logs, '--bo')
        plt.xlabel('Training set size')
        plt.ylabel('Average Conditional Log Likelihood')
        plt.title('Average Conditional Log Likelihood vs Training set size')
        plt.savefig(home_dir + 'fig3.png')
        plt.show()
    return True


def main():
    global X_train, X_test
    X_train, X_test = [], []
    print("-----========= Assignment 3 =========-----")
    for i in range(1, 401):
        X_train.append(read_train(i))
    for i in range(1, 201):
        X_test.append(read_test(i))

    N = [50, 100, 150, 200, 250, 300, 350, 400]
    w0 = np.append(feature_params.flatten(), transition_params.flatten())
    print(f_x(w0, 50))
    w = df_x(w0, 50)
    wF = w[:3210].reshape(10, 321)
    wT = w[3210:].reshape(10, 10)
    print('=' * 40)
    print(wF + feature_grad)
    print('=' * 40)
    print(wT + transition_grad)

    # pool = mp.Pool(num_cores)
    #
    # ws = zip

    # ws = prob_1(N, True)
    # times = [82.76734, 273.12812, 714.44263, 931.21983, 1188.21828, 1355.29131, 1446.87792]
    times = [86.95778322219849, 245.9661407470703, 668.2270562648773, 887.1454260349274,
             1113.9696233272552, 1355.3142182826996, 1461.212928533554, 1542.617322683334]

    plt.plot(N, times, '--bo')
    plt.xlabel('Training set size')
    plt.ylabel('Training Time (in seconds)')
    plt.title('Time in seconds vs Training set size')
    plt.savefig(home_dir + 'fig1.png')
    plt.show()

    ws = []
    for n in N:
        arr = np.load(home_dir+'weights_'+str(n)+'.txt.npy')
        ws.append(arr)

    prob_2(ws, N, True)
    prob_3(ws, N, True)

    exit()
    return


if __name__ == "__main__":
    main()
