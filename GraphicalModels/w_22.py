import numpy as np
from scipy.optimize import minimize


def fx(x):
    return (1-x[0]**2)+100*(x[1]-x[0]**2)**2


def dfx(x):
    grad = np.zeros_like(x)
    grad[0] = 400*x[0]**3 - (400*x[1] - 2)*x[0] - 2
    grad[1] = 200*x[1] - 200*x[0]**2
    return grad


def main:
	# to maximize, we minimize the inverse(-ve) of the function
	x_init = [1.0, 0.0]
    opt_x = minimize(fx, x_init, jac=dfx, method='BFGS', options={'disp': True}, tol=1e-7)
    print('Maxima acheived at : ', opt_x.x)
    print('Maxima value at optimal solution : ', -fx(opt_x.x))
    exit()


if __name__ == "__main__":
	main