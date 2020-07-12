import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm
from scipy.optimize import minimize

states = [-1, 1]
grid_size = 30
# np.random.seed(0)

data_dir = 'D:/UMass/Spring20/688/Homework/Assignment4/Assignment4/Data/'
img_dir = 'D:/UMass/Spring20/688/Homework/Assignment4/Assignment4/figures/'
X_train, X_test = None, None


def func(s):
    return 1.0/(1.0 + np.exp(-1.0 * s))


def find_nbr(i, j, idx):
    nbr = list()
    if i - 1 >= 0:
        nbr.append((i - 1, j))
    if i + 1 < idx[0]:
        nbr.append((i + 1, j))
    if j - 1 >= 0:
        nbr.append((i, j - 1))
    if j + 1 < idx[1]:
        nbr.append((i, j + 1))
    return nbr


def cond_prob_i(y, i, j, params):
    b, w = params
    nbr = find_nbr(i, j, w.shape)
    val = 0.0
    for m in nbr:
        val += w[m[0]][m[1]] * y[m[0]][m[1]]
    val += b[i][j]
    val *= 2.0
    return func(val)


def gibbs_sampling(t_max, params, x_init):
    b, w = params
    d = w.shape
    x_samp = list()
    mean = list()
    if x_init is None:
        x_samp.append(np.ones_like(w))  # init x[0] with ones
    else:
        x_samp.append(x_init)  # init with sample
    for t in range(1, 1 + t_max):
        x = np.array(x_samp[-1]).copy()
        for i in range(d[0]):
            for j in range(d[1]):
                prob = cond_prob_i(x, i, j, params)
                if prob > np.random.uniform():
                    x[i][j] = 1.0
                else:
                    x[i][j] = -1.0
        x_samp.append(np.array(x))
        mean.append(np.mean(x))
    return np.array(x_samp[1:]), np.array(mean)


def plot_vec(y, label, fname):
    plt.imshow(y, cmap='Greys_r', interpolation='nearest')
    plt.title(label)
    plt.savefig(img_dir + fname)
    plt.show()
    return True


def prob_3(flag=False):
    w_val_list = np.arange(0, 0.6, 0.1)
    b = np.zeros((grid_size, grid_size))
    w = np.ones((grid_size, grid_size))
    samples = list()
    means = list()
    for w_val in w_val_list:
        samp, mean = gibbs_sampling(100, (b, w_val*w), None)
        samples.append(samp)
        means.append(mean)
    if flag:
        for i in range(len(w_val_list)):
            plot_vec(samples[i][-1], 'w = '+str(w_val_list[i]), 'p3_'+str(i)+'.png')

    return np.array(samples), np.array(means)


def prob_5():
    n_trials = 100
    tot_mean = list()
    for n in tqdm(range(n_trials)):
        samps, means = prob_3()
        tot_mean.append(means)
    tot_mean = np.array(tot_mean)
    avg_means = np.mean(tot_mean, axis=0)
    plt.figure()
    plt.plot(avg_means[0, :], label='w = 0')
    plt.plot(avg_means[1, :], label='w = 0.1')
    plt.plot(avg_means[2, :], label='w = 0.2')
    plt.plot(avg_means[3, :], label='w = 0.3')
    plt.plot(avg_means[4, :], label='w = 0.4')
    plt.plot(avg_means[5, :], label='w = 0.5')
    plt.xlabel("No of iterations")
    plt.ylabel("Mean of y")
    plt.legend()
    plt.savefig(img_dir+'p5.png')
    plt.show()
    return True


def rescale_vec(x):
    return np.interp(x, (x.min(), x.max()), (-1, 1))


def prob_7():
    x = rescale_vec(plt.imread(data_dir+'im_noisy.png'))
    x_test = rescale_vec(plt.imread(data_dir+'im_clean.png'))
    plot_vec(x, 'noisy_img', 'q7_test.png')
    b = 0.5 * x
    w = 0.3 * np.ones_like(x)
    samples, means = gibbs_sampling(100, (b, w), x)
    sample_img = samples.mean(axis=0)
    sample_img = rescale_vec(sample_img)
    plot_vec(sample_img, 'b = 0.5x ; w = 0.3', 'q7.png')
    error = np.abs(sample_img - x_test)
    print("Mean of img : ", np.mean(sample_img))
    print("Mean error : ", np.mean(error))
    return True


def obj_func(params, flag=False):
    global X_train, X_test
    b = params[0] * X_train
    w = params[1] * np.ones_like(X_train)
    outputs, _ = gibbs_sampling(100, (b, w), X_train)
    op_mean = rescale_vec(outputs.mean(axis=0))
    if flag:
        plot_vec(op_mean, 'best_img', 'best_opt.png')
    error = np.mean(np.abs(op_mean - X_test))
    print(error)
    return error


def prob_8():  # Optimizer method
    global X_train, X_test
    X_train = rescale_vec(plt.imread(data_dir+'im_noisy.png'))
    X_test = rescale_vec(plt.imread(data_dir+'im_clean.png'))
    print("Searching for best params...")
    best_params = minimize(obj_func, np.array([0.5, 0.5]), method='Nelder-Mead', tol=1e-3,
                           options={'disp': True, 'maxiter': 1000, 'fatol': 1e-3})
    print(best_params.x)
    params = best_params.x
    # params = [0.7, 0.9999999999999999]  # check for value
    # params = [0.7, 3.5]
    err = obj_func(params, False)
    print("Error at best params : ", err)
    return True


def prob_8_1():  # Grid search method
    global X_train, X_test
    X_train = rescale_vec(plt.imread(data_dir + 'im_noisy.png'))
    X_test = rescale_vec(plt.imread(data_dir + 'im_clean.png'))
    best_params = None
    best_err = 100.0
    for b in np.arange(0.5, 1.1, 0.1):  # Tuned settings
        for w in np.arange(2.0, 3.5, 0.5):
            err = obj_func((b, w))
            if err < best_err:
                best_params = [b, w]
                best_err = err
    print(best_params)
    print("Error at best params : ", best_err)
    return True


def main():
    print("Running CS688_hw4 ...")
    prob_3(True)
    prob_5()
    prob_7()
    prob_8()  # Optimizer method
    prob_8_1()  # Grid search method
    exit(0)
    return True


if __name__ == "__main__":
    main()
