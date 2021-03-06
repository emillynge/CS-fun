__author__ = 'emil'
import requests
import numpy as np
from matplotlib import pyplot as plt
from collections import namedtuple
from matplotlib import gridspec
from scipy.optimize import curve_fit, fmin_tnc, check_grad, approx_fprime

redownload_data = False

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

# Remove data above and below t_range. Remove data points with NaN values
cropped_data = raw_data[t_range[0] <= raw_data[:, 0], :]
cropped_data = cropped_data[t_range[1] >= cropped_data[:, 0], :]
no_nan = np.logical_not(np.logical_or(np.isnan(cropped_data[:, 1]), np.isnan(cropped_data[:, 2])))
cropped_data = cropped_data[no_nan, :]


def optfun_generator(y, X, *poly_lines):
    """
    Generate optimizing function
    :param y: ground truth iterable
    :param X: x-values corresponding to ground truth data points
    :param poly_lines: instances of Polyline objects containing degree of polynomium and initial guess of cutoff
    :return:
        theta0: randomized initial parameters
        optfun: optimizing function to be passed to minimizing function
    """
    poly_orders = tuple(line.order for line in poly_lines)
    n_params = sum(poly_orders) + 2 * len(poly_orders)

    def optfun(theta):
        y_hat, dy_dx, x = np.zeros((1, len(X))), np.zeros((1, len(X))), np.matrix(X).reshape((1, len(X)))
        grad = list(range(n_params))
        i = 0
        for order in poly_orders:
            p = theta[i:i+order].tolist()
            p.reverse()
            lineval = np.polyval(p, x)                                  # actual polyline
            on_logi = (1 / (1 + np.exp(theta[i+order] * x + theta[i + order + 1])))   # on function
            #polyval_contribution *= (1 / (1 + np.exp(theta[i+order+2] * x + theta[i + order + 3])))   # off function

            _dy_dx = np.zeros(dy_dx.shape)
            for degree in range(order):
                if degree > 0:
                    _dy_dx += theta[i+degree]*np.power(x, degree-1)*(degree)
                grad[i+degree] = np.multiply(np.power(x, degree), on_logi)
            _dy_dx -= np.multiply(_dy_dx, on_logi)
            _dy_dx -= np.multiply(np.multiply(lineval, on_logi), (1 - on_logi)) * grad[i+degree+1]
            grad[i+degree+1] = -np.multiply(np.multiply(np.multiply(lineval, on_logi), 1 - on_logi), x)
            grad[i+degree+2] = -np.multiply(np.multiply(lineval, on_logi), 1 - on_logi)
            dy_dx += _dy_dx

            y_hat += np.multiply(lineval, on_logi)
            i += order + 2 # + 4

        residual = y_hat - y
        grad = [np.sum(np.multiply(g, 2 * residual)) for g in grad]
        err = residual @ residual.T
        return err[0,0], grad, y_hat.tolist()[0], dy_dx.tolist()[0] #TODO split function. it's doing way too many things!

    def _optfun(theta):
        """
        wrapper of optimizing function so it fits interface for minimizing functions
        """
        return optfun(theta)[:2]

    # initialize weight vector theta
    theta0 = list()
    for order, cutoff in poly_lines:
        theta0.append(np.random.randn(order))   # polynomial coefficients -> normal random numbers
        theta0.append(np.ones((1,)))            # slope of cutoff -> 1
        theta0.append(-np.ones((1,))*cutoff)    # cutoff parameter -> -1 * coordinate of desired cutoff
    theta0 = np.concatenate(tuple(theta0))      # collapse to a single numpy array
    return theta0, optfun, _optfun

gs = gridspec.GridSpec(2, 2)                    # used for subplotting
resolution = 5                                  # points per second
X_grid = np.linspace(t_range[0], t_range[1], (t_range[1]-t_range[0]) * resolution)    # uniform x-grid to use for interpolation
main_legend = dict()
main_ax = plt.subplot(gs[:,0])
# declaration of PolyLine object to hold information:
#   order - degree of polynomial for line segment
#   cutoff - initial x-coordinate for when to activate line segment
Polyline = namedtuple('PolyLine', 'order cutoff')


def fit(name, idx, color, *polylines, recompute_theta=False):
    """
    Fit polyline segments to data in cropped_data[:, idx]. plot using color and name
    :param name: name of fit
    :param idx: index of data to be fitted
    :param color: color of fit
    :param polylines: PolyLine objects
    :param recompute_theta: bool - controls whether theta should be recomputed or loaded from file
    :return: None
    """
    # functions to be used for fitting
    theta0, optfun, _optfun = optfun_generator(cropped_data[:, idx], cropped_data[:, 0], *polylines)

    if not recompute_theta:
        popt = np.load('theta{}.npy'.format(name))    # load previously saved theta parameters and use for fitting
    else:
        popt, *_ = fmin_tnc(_optfun, theta0, maxfun=10**4)
        np.save('theta{}.npy'.format(name), popt)

    # functions to be used for interpolating onto X_grid
    _, optfun, _optfun = optfun_generator(X_grid, X_grid, *polylines)
    _, __, y_hat, dy_dx = optfun(popt)

    dy_dx = np.diff(np.max(y_hat) - y_hat)
    _dy_dx = np.array(dy_dx)
    main_ax.plot(cropped_data[:, 0], cropped_data[:, idx], color + 'o', ms=1)
    main_legend[name] = main_ax.plot(X_grid, y_hat, color, linewidth=2, label=name)
    plt.ylabel('Remaining mass [kg]')
    plt.xlabel('time')

    plt.subplot(2,2,2)
    plt.plot(X_grid[1:], _dy_dx * resolution, color + '--', linewidth=1)
    plt.ylabel('Mass consumption [kg/s]')

    return np.array(y_hat), np.array(dy_dx)

fuel_y, fuel_diff =fit('fuel', 1, 'r', Polyline(2, 2), Polyline(2, 5), Polyline(2, 7), Polyline(2, 10), Polyline(2, 12))
oxidizer_y, oxidizer_diff =fit('oxidizer', 2, 'g', Polyline(2, 1), Polyline(2, 1.5), Polyline(2, 2.8), Polyline(2, 5.5),
                               Polyline(2, 12.5), Polyline(2, 13), recompute_theta=False)

OF = np.divide(oxidizer_diff, fuel_diff)

# Negative mass consumption is a measurement artifact and is therefore corrected to 0
oxidizer_diff_neg_only = oxidizer_diff.copy()
oxidizer_diff_neg_only[oxidizer_diff < 0] = 0
fuel_diff_neg_only = fuel_diff.copy()
fuel_diff_neg_only[fuel_diff < 0] = 0
OF_corrected = np.divide(oxidizer_diff_neg_only, fuel_diff_neg_only)

plt.subplot(2,2,4)
plt.plot(X_grid[1:], OF, 'b')
plt.plot(X_grid[1:], OF_corrected, 'm--', linewidth=3)
plt.legend(('Raw', 'Corrected'))
plt.ylabel('O/F ratio')
plt.xlabel('time')
plt.ylim([-1, 5])


main_ax.legend()
#main_ax.legend(main_legend.values(), main_legend.keys())
plt.show()
