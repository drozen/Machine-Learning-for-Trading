"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.lines as lines

if __name__=="__main__":
    #inf = open('Data/ripple.csv')
    #inf = open('Data/3_groups.csv')
    #inf = open('Data/simple.csv')
    inf = open('Data/best4linreg.csv')
    #inf = open('Data/best4KNN.csv')


    print 'LinReg Learner'
    print inf.name
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = math.floor(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data

    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

     # random samplings
    index = np.random.choice(range(len(data)), size = len(data), replace = False)
    trainIndex = sorted(index[:train_rows])
    testIndex = sorted(index[train_rows:])
    trainX = data[trainIndex,0:-1]
    trainY = data[trainIndex,-1]

    #print 'data[trainIndex]:\n', data[trainIndex]
    # print "trainX: \n ", trainY
    # print "trainY: \n ", trainY

    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]
    #print "testY: \n ", testY

    # random samplings
    testX = data[testIndex,0:-1]
    testY = data[testIndex,-1]


    # create a learner and train it
    learner = lrl.LinRegLearner() # create a LinRegLearner
    learner.addEvidence(trainX, trainY) # train it

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]

 # Plot data
    fig = plt.figure()
   # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')

    x1 = testX[:,0]
    x2 = testX[:,1]
    y = testY

    #ax.plot(x1, x2, y, zs=0, zdir='y', label='zs=0, zdir=z')

    #colors = ('r', 'g', 'b', 'k')
    ax.scatter(x1, x2, y, zdir='z', c='b', label = 'Test Y')

    x1 = testX[:,0]
    x2 = testX[:,1]
    y = predY
    ax.scatter(x1, x2, y, zdir='z', c='y', label = 'Pred Y')

    ax.legend()
  # for sinecurve
  #   ax.set_xlim3d(0, 6.3)
  #   ax.set_ylim3d(0, 6.3)
  #   ax.set_zlim3d(-1, 1)

    #    # for line
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)

    # # for simple.csv
    # ax.set_xlim3d(0,11)
    # ax.set_ylim3d(0,11)
    # ax.set_zlim3d(0,11)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    plt.title('LinReg on best4linreg data')
    plt.show()