# Create your own dataset generating code (call it best4KNN.py) that creates data that performs significantly better with KNNLearner than LinRegLearner. Explain your data generating algorithm, and explain why KNNLearner performs better. Your data should include at least 2 dimensions in X, and at least 1000 points. (Don't use bagging for this section).

# output data in the same format as the provided example files.

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
import math

if __name__=="__main__":


   # generate data
    x1 = np.linspace(0,6.30,1000)
    x2 = x1

    y = np.sin(x1)

    output = np.zeros((1000,3))
    output[:,0] = x1
    output[:,1] = x2
    output[:,2] = y
    # output = np.concatenate((x,y), axis = 1)

    #print 'output:\n', output
    csvOutput = []
    for a, b, c in output:
        string = str(a)+','+str(b)+','+str(c)
        csvOutput.append(string)
    # convert ouptut to txt file
    b = open('Data/best4KNN.csv', 'w')
    for out in csvOutput:
        b.write(out)
        b.write('\n')
    b.close()

    # Plot data
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax = fig.add_subplot(111, projection='3d')

    #ax.plot(x1, x2, y, zs=0, zdir='y', label='zs=0, zdir=z')

    #colors = ('r', 'g', 'b', 'k')
    ax.scatter(x1, x2, y, zdir='z', c='b')

    ax.legend()
    ax.set_xlim3d(0, 6.3)
    ax.set_ylim3d(0, 6.3)
    ax.set_zlim3d(-1, 1)

    # # for simple.csv
    # ax.set_xlim3d(0,11)
    # ax.set_ylim3d(0,11)
    # ax.set_zlim3d(0,11)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    plt.show()