"""
    LINEAR CRF MODELLING - SEQUENCE LABELING
"""

import numpy as np
import itertools as itr
from collections import OrderedDict
from scipy.optimize import minimize

labels = 'etainoshrd'
labels_dict = {labels.index(c): c for c in labels}
labels_pos = {c: labels.index(c) for c in labels}
home_dir = 'D:\\UMass\\Spring20\\688\\Homework\\Assignment2\\Assignment2/'

model = OrderedDict()


def read_test(num):
    vec = np.loadtxt(home_dir+'data/test_img'+str(num)+'.txt')
    return vec


def read_train(num):
    vec = np.loadtxt(home_dir+'data/train_img'+str(num)+'.txt')
    return vec


def init():
    feature_params = np.loadtxt(home_dir+'model/feature-params.txt')
    transition_params = np.loadtxt(home_dir+'model/transition-params.txt')
    test_labels = np.loadtxt(home_dir+'data/test_words.txt', dtype=str)
    train_labels = np.loadtxt(home_dir+'data/train_words.txt', dtype=str)

    model[1] = [2]
    model[2] = [3]
    model[3] = [4]
    model[4] = []

    return feature_params, transition_params, test_labels, train_labels


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
        idx_1 = labels_pos[y_true[j+1]] if j < L-1 else -1
        if debug:
            print(j, ' | ', y_true[j], ' | ', idx, ' | ', idx_1)
        if j == L-1:
            energy += np.log(pots[j][idx])
        if j < L-1:
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
        prob = np.exp(energy_func(s, x, wf, wt))/part
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
    (wf, wt, test_labels, train_labels) = params
    L, F = x.shape
    x_pots = potential(x, wf)
    # forward pass
    forward_messages = []
    z_fwd = 0.0
    for i in range(L):
        if i == 0:
            forward_messages.append(x_pots[i].dot(np.exp(wt)))
        elif i == L-1:
            forward_messages.append(x_pots[i] * forward_messages[i - 1])
            z_fwd = x_pots[i].dot(forward_messages[i-1].T)
        else:
            forward_messages.append((x_pots[i] * forward_messages[i-1]).dot(np.exp(wt)))

    forward_messages = np.array(forward_messages)

    # backward pass
    backward_messages = np.array([np.zeros(wt.shape[0])]*forward_messages.shape[0])
    z_bck = 0.0
    for i in range(L-1, -1, -1):
        if i == L-1:
            backward_messages[i,:] = x_pots[i].dot(np.exp(wt))
        elif i == 0:
            backward_messages[i,:] = x_pots[i] * backward_messages[i+1]
            z_bck = x_pots[i].dot(backward_messages[i+1].T)
        else:
            backward_messages[i,:] = (x_pots[i] * backward_messages[i+1]).dot(np.exp(wt))
    backward_messages = np.array(backward_messages)

    return forward_messages[:-1], z_fwd, backward_messages[1:], z_bck, x_pots


def marginals(f_msg, b_msg, z, pots, params):
    (wf, wt, test_labels, train_labels) = params
    L = pots.shape[0]

    # marginal prob
    marginal_p = []
    for i in range(L):
        if i == 0:
            marginal_p.append(pots[i]*b_msg[i]/z)
        elif i == L - 1:
            marginal_p.append(f_msg[i-1]*pots[i]/z)
        else:
            marginal_p.append(f_msg[i-1]*pots[i]*b_msg[i]/z)

    # print(f_msg.shape, b_msg.shape,pots.shape)
    # pairwise prob
    pairwise_p = []
    for i in range(L-1):
        pm = np.zeros((10, 10))
        for j in range(10):
            for k in range(10):
                if i == 0:
                    pm[j][k] = b_msg[i+1][k] * np.exp(wt)[j][k] * (1/z) * pots[i][j] * pots[i + 1][k]
                elif i == L-2:
                    pm[j][k] = f_msg[i-1][j] * np.exp(wt)[j][k] * (1 / z) * pots[i][j] * pots[i + 1][k]
                else:
                    pm[j][k] = f_msg[i-1][j] * b_msg[i+1][k] * np.exp(wt)[j][k] * (1 / z) * pots[i][j] * pots[i + 1][k]
        pairwise_p.append(pm)

    return np.array(marginal_p), np.array(pairwise_p)


def prob_1():
    print('Running for Question 1......')
    wf, wt, test_labels, train_labels = init()
    print('Question 1.1 : ')
    # 1.1
    potentials = np.log(potential(read_test(1), wf, False))
    print(potentials)

    # 1.2
    print('Question 1.2 :')
    for i in range(1, 4):
        energy_word = energy_func(test_labels[i-1], read_test(i), wf, wt, False)
        print(test_labels[i-1], ' | ', energy_word)

    # 1.3
    print('Question 1.3 : ')
    x_set = []
    lmt = 3
    for i in range(1, lmt+1):
        x_set.append(read_test(i))
    x_set = np.array(x_set)
    log_part = partition(x_set, wf, wt, True)

    # 1.4
    print('Question 1.4 : ')
    joint_prob = []
    for i in range(lmt):
        word_p, p, prb = joint_probability(x_set[i], wf, wt, log_part[i])
        joint_prob.append(prb)
        print(word_p, ' | ', p)

    # 1.5
    print('Question 1.5 : ')
    marginal_prob(joint_prob[0], test_labels[0])

    print('Done')
    return True


def prob_2():
    # 2.1
    wf, wt, test_labels, train_labels = init()
    word_idx = 1
    params = (wf, wt, test_labels, train_labels)
    print('Question 2.1 : ')
    fwd_msg, zf, bck_msg, zb, x_pot = message_passing(read_test(word_idx), params)
    print(np.log(fwd_msg))
    print(np.log(zf))
    print(np.log(bck_msg))
    print(np.log(zb))

    # 2.2
    print('Question 2.2 : ')
    prob_m, prob_p = marginals(fwd_msg, bck_msg, zf, x_pot, params)
    print(prob_m)
    print('-'*100)
    print(prob_p)

    print('3 x 3 tables : ')
    chas = ['t', 'a', 'h']
    chas_idx = [labels_pos[c] for c in chas]
    for p in prob_p:
        print(p[chas_idx][:, chas_idx])
        print('-'*50)

    # 2.3
    print('Question 2.3 : ')
    pred_words = []
    for i in range(1, len(test_labels)+1):
        f_msg, fz, b_msg, bz, xpot = message_passing(read_test(i), params)
        prob_s, prob_m = marginals(f_msg, b_msg, fz, xpot, params)
        pred_words.append([labels_dict[idx] for idx in np.argmax(prob_s, axis=1)])

    true_words = [list(ch) for ch in test_labels]

    print(true_words[:5])
    print(pred_words[:5])

    corr = 0
    tot = 0
    for k in range(len(test_labels)):
        tot += len(test_labels[k])
        for l in range(len(test_labels[k])):
            if test_labels[k][l] == pred_words[k][l]:
                corr += 1

    acc = (corr/tot)*100.0
    print('Accuracy obtained : ', acc, ' %')

    return True


def prob_3():
    # 3.5
    print('Question 3.5 : ')
    wf, wt, test_labels, train_labels = init()
    N = 50
    params = (wf, wt, test_labels, train_labels)
    log_lh = 0.0
    for i in range(1, N+1):
        f_msg, fz, b_msg, bz, xpot = message_passing(read_train(i), params)
        prob_s, prob_m = marginals(f_msg, b_msg, fz, xpot, params)
        idx_y = [labels_pos[u] for u in list(train_labels[i-1])]
        lh = 0.0
        for m in range(prob_s.shape[0]):
            lh += np.log(prob_s[m][idx_y[m]])
        log_lh += lh
    log_lh /= N
    # log_lh *= -1
    print('Avg Log Likelihood : ', log_lh)

    return True


def fx(x):
    return (1-x[0]**2)+100*(x[1]-x[0]**2)**2


def dfx(x):
    grad = np.zeros_like(x)
    grad[0] = 400*x[0]**3 - 400*x[1]*x[0] + 2*x[0] - 2
    grad[1] = 200*x[1] - 200*x[0]**2
    return grad


def prob_4():
    print('Question 4.2 : ')
    x_init = np.array([0.1, 0.1])
    opt_x = minimize(fx, x_init, jac=dfx, method='BFGS', options={'disp': True, 'gtol': 1e-7}, tol=1e-8)
    print('Maxima acheived at : ', opt_x.x)
    print('Maximum value at optimal solution : ', -fx(opt_x.x))

    return True


def main():
    print("Running Assignment 2........")
    # prob_1()
    prob_2()
    # prob_3()
    # prob_4()
    exit()
    return


if __name__ == "__main__":
    main()