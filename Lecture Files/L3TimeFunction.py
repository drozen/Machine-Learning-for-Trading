__author__ = 'Daniel Rozen'

import time


def test_run():
    t1 = time.time()
    print "ML4T"
    t2 = time.time()
    print "The time taken is ", t2-t1, " secs"

if __name__ == "__main__":
    test_run()
