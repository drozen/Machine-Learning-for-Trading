__author__ = 'Daniel Rozen'
"""Fit a line to a given set of data points using optimization."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

def error_poly(C, data):
    """ Compute error between given polynomial and observed data
    :param C: numpy.poly1d object or equivalent array representing polynomial coefficients
    :param data: 2D aray where each row is a point (x,y)
    :return: error as a single real value.
    """

    # Metric: sum of squared Y-axis differneces
    err = np.sum((data[:,1] - np.polyval(C, data[:,0])) ** 2)
    return err


def error(line, data): # error function
    """ Compute error between given line model and observed data.

    Parameters
    __________
    line: tuple/list/array (C0, C1) where C0 is slope and C1 is Y-intercept
    data: 2D array where each row is a point (x,y)

    Returns error as a single real value
    """
    # Metric: Sum of squared Y-axis differences
    err = np.sum((data[:, 1] - (line[0] * data[:, 0] + line[1])) ** 2)
    return err

def test_run():
    # define original line
    l_orig = np.float32([4,2])
    print "Original line: C0 = {}, C1 = {}".format(l_orig[0],  l_orig[1])
    Xorig = np.linspace(0,10,21)
    Yorig = l_orig[0] * Xorig + l_orig[1]
    plt.plot(Xorig, Yorig, 'b--', linewidth = 2.0, label = "Original line")

    # Generate noisy data points
    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:,0], data[:,1], 'go', label="Data points")

    # Try to fit a line to this data
    l_fit = fit_line(data, error)
    print "Fitted line: C0 = {}, C1".format(l_fit[0], l_fit[1])
    plt.plot(data[:, 0], l_fit[0] * data[:,0] + l_fit[1], 'r--', linewidth=2.0, label = 'Line Fit')

    # Plot function values, mark minima
    Xplot = np.linspace(0.5,2.5,21)
    Yplot = f(Xplot)

    plt.show()

def fit_poly(data, error_func, degree=3):
    """Fit a polynomial to given data, using supplied error function
    :param data: 2D array where each row is a point (x, y)
    :param error_func: FUNCTION THAT COMPUTES THE ERROR BETWEEN A polynomial AND OBSERVED DEATA

    :return: polynomial that minimizes the error function
    """

    # Generate initial guess for line model
    Cguess = np.poly1d(np.ones(degree + 1, dtype=np.float32)) # set coefficients to 1


    # plot initial guess (optional)
    x = np.linspace(-5,5,21)
    plt.plot(x, np.plyval(Cguess,x), 'm--', linewidth = 2.0, label="Initial guess")

    # Call optimizer to minimize error function - THIS IS THE KEY
        # args=(data,) is a way we can pass the error function
    result = spo.minimize(error_func, Cguess, args=(data,), method='SLSQP',   options = {'disp': True})
    return np.poly1d(result.x)

if __name__ == "__main__":
    test_run()

