__author__ = 'Daniel Rozen'
import numpy as np

def test_run():
    a = np.random.rand(5,4)
    print "Array:\n", a

    # accessing element at post (3,2)
    element = a[3,2]
    print '\n element:\n', element

    # elements in defined range
    print '\n a[0, 1:3]\n', a[0, 1:3]
    print '\n a[0:2,0:2]\n', a[0:2,0:2]

    # Slicing
    print a[:,0:3:2] # will select columns 0, 2 for every row (choosing every 2nd column as indicated by the last 2)

    # assign a single value to an entire row
    a[0,:] = 2
    print a

    # assign a list to a column in an arr
    a[:,3] = [1,2,3,4,5]
    print a

    # # accessing elements
    #
    # a = np.random.rand(5)
    # print a
    # # accessing using list of indices
    # indices = np.array([1,1,2,3])
    # print a[indices]
    #
    # # calc mean
    # mean = a.mean()
    # print mean
    # # masking
    # print a[a<mean]
    #
    # a[a<mean] = mean
    # print a

if __name__ == "__main__":
    test_run()
