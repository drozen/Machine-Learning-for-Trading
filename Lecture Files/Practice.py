__author__ = 'Daniel Rozen'
import numpy as np
A = np.ones((3,3))
w = np.array([0.0, 0.1, 0.2])
print A*w
print (A*w).sum()
print (A*w).sum(axis=0)
print (A*w).sum(axis=1)