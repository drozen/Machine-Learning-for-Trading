import BagLearner as bl
import KNNLearner as knn

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.lines as lines

import numpy as np
import math

if __name__=="__main__":

    inf = open('Data/ripple.csv')
    #inf = open('Data/3_groups.csv')
    #inf = open('Data/simple.csv')
    #inf = open('Data/best4linreg.csv')
    #inf = open('Data/best4KNN.csv')

    print 'KNN Learner'
    print inf.name

    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])
    #print data

    # compute how much of the data is training and testing
    train_rows = int(math.floor(0.6* data.shape[0]))
    test_rows = int(data.shape[0] - train_rows)

    # # separate out training and testing data
    # trainX = data[:train_rows,0:-1]
    # trainY = data[:train_rows,-1]

    # random samplings
    index = np.random.choice(range(len(data)), size = len(data), replace = False)
    # trainIndex = sorted(index[:train_rows])
    # testIndex = sorted(index[train_rows:])
    trainIndex = sorted(index[:train_rows])
    testIndex = sorted(index[train_rows:])

    trainX = data[trainIndex,0:-1]
    trainY = data[trainIndex,-1]

    # testX = data[train_rows:,0:-1]
    # testY = data[train_rows:,-1]

    # random samplings
    testX = data[testIndex,0:-1]
    testY = data[testIndex,-1]

    # create a learner and train it
    inSampleRMSE = [0]
    inSampleCorr = [0]
    outSampleRMSE = [0]
    outSampleCorr = [0]

    ##loop through k values
    # for k in range(3,10):
    # learner = bl.BagLearner(learner = knn.KNNLearner, kwargs = {"k":k}, bags = bags, boost = False)

    k=3
    kwargs = {"k":k}



    # loop through bag values

    #bags = 1
    ##loop through k values

    bagsRange = range(1,11)
    for bags in bagsRange:
    # learner = bl.BagLearner(learner = knn.KNNLearner, kwargs = {"k":k}, bags = bags, boost = False)

        learner = bl.BagLearner(learner = knn.KNNLearner, kwargs = kwargs, bags = bags, boost = False)
        learner.addEvidence(trainX, trainY)

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        print "Number of bags: ", bags
        print "In sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=trainY)
        print "corr: ", c[0,1]
    
        inSampleRMSE.append(rmse)
        inSampleCorr.append(c[0,1])
    
        # evaluate out of sample
        predY = learner.query(testX) # get the predictions
        rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
        print
        print "Out of sample results"
        print "RMSE: ", rmse
        c = np.corrcoef(predY, y=testY)
        print "corr: ", c[0,1]
    
        outSampleRMSE.append(rmse)
        outSampleCorr.append(c[0,1])

    print 'Summary results:'
    n=1
    # bagsRange = (range(1,10,2) + range(10,55,15) + 155)

    for bags in bagsRange:
        
        print 'bags =', bags, ' inRMSE=',inSampleRMSE[n], ' inCorr=',inSampleCorr[n],  ' outRMSE=',outSampleRMSE[n], ' outCorr=', outSampleCorr[n]
        n+=1

    #plot knn data

    # Plot RMSE vs k
  
    bags = bagsRange
  #print 'len bags ', len(bags)
    y1 = inSampleRMSE[1:]
  #print 'len y1 ',len(y1)
    y2 = outSampleRMSE[1:]
  
    plt.plot(bags, y1, 'ro-', label = 'In Sample RMSE')
    plt.plot(bags, y2, 'go-', label ='Out of Sample RMSE')
    # plt.plot(bags, y1,  'r', label ='in sample RMSE', bags, y2, 'g',label='out sample RMSE')
    plt.xlabel('Bags')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Bags KNN with Ripple data')
    plt.show()
    #plot Corr vs bags
    y1 = inSampleCorr[1:]
    y2 = outSampleCorr[1:]
    plt.plot(bags, y1, 'ro-', label ='In Sample Correlation')
    plt.plot(bags, y2, 'go-', label ='Out of Sample Correlation')
    # plt.plot(bags, y1,  'r', label ='in sample Corr', bags, y2, 'g',label='out sample Corr')
    plt.xlabel('Bags')
    plt.ylabel('Corr')
    plt.title('Correlation vs Bags KNN with Ripple data')

    plt.show()

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
    predY = learner.query(testX)
    y = predY
    ax.scatter(x1, x2, y, zdir='z', c='y', label = 'Predicted Y')

    ax.legend()

    # # for sinecurve
    # ax.set_xlim3d(0, 6.3)
    # ax.set_ylim3d(0, 6.3)
    # ax.set_zlim3d(-1, 1)

    # for ripple
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    # # for best4linreg
    # ax.set_xlim3d(0,1)
    # ax.set_ylim3d(0,1)
    # ax.set_zlim3d(0,1)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    plt.title('BagLearner with KNN with ripple data')
    plt.show()


