__author__ = 'Daniel Rozen'

""" Creating NumPY arrays."""

import numpy as np

def test_run():
    # # List to 1D array     nd array means: n-dimensional array
    # print np.array([2,3,4])
    # # List of tuples to 2D array
    # print np.array([(2,3,4), (5,6,7)])

    # Empty array
    # print np.empty(5)
    # print np.empty((5,4,3)) # 3D array

    # # Array of 1s
    # print np.ones((5,4), dtype=np.int_)

    # Generate an array full of random numbers, uniformly sampled from [0.0. 1.0)
    #print np.random.rand(5,4) # function arguments (not a tuple)

    # Sample numbers from a Guassian (normal) distribution
    #print np.random.normal(size=(2,3)) # standard normal (mean = 0 , s.d. = 1)
    # print np.random.normal(50,10, size=(2,3)) # mean = 50, sd = 10

    # ARRAY ATTRIBUTES:

    # a = np.random.random((5,4))
    # print a
    # print a.dtype

    # OPERATIONS ON ARRAYS

    np.random.seed(693) # seed the random number generator - in order to get the same sequence of numbers every time
    a = np.random.randint(0, 10, size=(5,4)) # 5x4 random integers in [0,10)
    print "Array:\n", a

    print "Sum of all elements:", a.sum()

    # Iterage over rows, to compute sum of each column
    print "Sum of each column:\n", a.sum(axis=0)

    #vice versa
    print "Sum of each row:\n", a.sum(axis=1)
    print "Min of each col:", a.min(axis=0)
    print "Max of each row:", a.max(axis=1)
    print "mean of all elements:", a.mean() # leave out axis argument

    # Satitstics: min, max, mean (across rows, cols, and overall)


if __name__ == "__main__":
    test_run()

