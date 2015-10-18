__author__ = 'emil'
import requests
import numpy as np
from matplotlib import pyplot as plt
from collections import  namedtuple
from scipy.optimize import curve_fit, fmin_tnc, check_grad, approx_fprime
np.random.seed(2)

redownload_data = False
recomp_theta = False
t_range = (0, 20)
data_url = "https://dl.dropboxusercontent.com/u/2640195/BPM5_data.txt"
def url2mat(url):
    r = requests.get(url)
    mat = np.fromiter((float(element) for line in r.text.split('\n') if '\t' in line for element in line.split('\t')),
                      float)
    return np.matrix(mat.reshape((len(mat)/3, 3)))
if redownload_data:
    raw_data = url2mat(data_url)
    np.save('rawdat.npy', raw_data)
else:
    raw_data = np.load('rawdat.npy')

cropped_data = raw_data[t_range[0] <= raw_data[:, 0], :]
cropped_data = cropped_data[t_range[1] >= cropped_data[:, 0], :]

def show_plots(data):
    plt.plot(data[:, 0], data[:, 1], 'r')
    plt.plot(data[:, 0], data[:, 2], 'g')
    plt.show()

#show_plots(cropped_data)


def optfun_generator(y, X, *poly_orders, cutoffs=None):
    n_params = sum(poly_orders) + 2 * len(poly_orders)

    def optfun(theta):
        y_hat, x = np.zeros((1, len(X))), np.matrix(X).reshape((1, len(X)))
        grad = list(range(n_params))
        i = 0
        for order in poly_orders:
            p = theta[i:i+order].tolist()
            p.reverse()
            lineval = np.polyval(p, x)                                  # actual polyline
            on_logi = (1 / (1 + np.exp(theta[i+order] * x + theta[i + order + 1])))   # on function
            #polyval_contribution *= (1 / (1 + np.exp(theta[i+order+2] * x + theta[i + order + 3])))   # off function


            for degree in range(order):
                grad[i+degree] = np.multiply(np.power(x, degree), on_logi)
            grad[i+degree+1] = -np.multiply(np.multiply(np.multiply(lineval, on_logi), 1 - on_logi), x)
            grad[i+degree+2] = -np.multiply(np.multiply(lineval, on_logi), 1 - on_logi)

            y_hat += np.multiply(lineval, on_logi)
            i += order + 2 # + 4

        residual = y_hat - y
        grad = [np.sum(np.multiply(g, 2 * residual)) for g in grad]
        err = residual @ residual.T
        return err[0,0], grad, y_hat.tolist()[0]

    def _optfun(theta):
        return optfun(theta)[:2]

    x0 = list()
    for order,cutoff in zip(poly_orders, cutoffs):
        x0.append(np.random.randn(order))
        x0.append(np.ones((1,)))
        x0.append(-np.ones((1,))*cutoff)
    x0 = np.concatenate(tuple(x0))
    return x0, optfun, _optfun


def fit(name, idx, color):
    no_nan_fuel = np.logical_not(np.isnan(cropped_data[:, idx]))
    theta0, optfun, _optfun = optfun_generator(cropped_data[no_nan_fuel, idx], cropped_data[no_nan_fuel, 0], 2,2,2,2,2,
                                               cutoffs=(5, 7, 9, 12,15))

    if not recomp_theta:
        theta0 = np.load('theta{}.npy'.format(name))
    popt, *_ = fmin_tnc(_optfun, theta0, maxfun=10**4)
    np.save('theta{}.npy'.format(name), popt)
    _, __, y_hat = optfun(popt)

    plt.plot(cropped_data[:, 0], cropped_data[:, idx], color + 'o', ms=1)
    plt.plot(cropped_data[no_nan_fuel, 0], y_hat, color, linewidth=2)

fit('fuel', 1, 'r')
fit('oxidizer', 2, 'g')
plt.show()
