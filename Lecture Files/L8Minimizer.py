__author__ = 'Daniel Rozen'
"""Minimize an objective function using SciPy."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo

def f(X):
    """Given a scalr X, return some value (a real number)."""
    Y = (X - 1.5)**2 + 0.5
    print "X = {}, Y = {}".format(X,Y) # for tracing
    return Y

# Fit a line to a given set of data points using optimization.
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
    Xguess = 2.0
    min_result = spo.minimize(f, Xguess, method = 'SLSQP', options = {'disp': True})
    print "Minima found at:"
    print "X = {}, Y = {}".format(min_result.x, min_result.fun) # for tracing

    # Plot function values, mark minima
    Xplot = np.linspace(0.5,2.5,21)
    Yplot = f(Xplot)
    plt.plot(Xplot, Yplot)
    plt.plot(min_result.x, min_result.fun, 'ro')
    plt.title("Minima of an objective function")
    plt.show()

if __name__ == "__main__":
    test_run()

