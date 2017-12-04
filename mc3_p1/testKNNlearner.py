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

    # compute how much of the data is training and testing
    train_rows = int(math.floor(0.6* data.shape[0]))
    test_rows = int(data.shape[0] - train_rows)

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]

    # random samplings
    index = np.random.choice(range(len(data)), size = len(data), replace = False)
    trainIndex = sorted(index[:train_rows])
    testIndex = sorted(index[train_rows:])
    # trainIndex = (index[:train_rows])
    # testIndex = (index[train_rows:])

    trainX = data[trainIndex,0:-1]
    trainY = data[trainIndex,-1]

    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]
    #print "testY: \n ", testY

    # random samplings
    testX = data[testIndex,0:-1]
    testY = data[testIndex,-1]

    # create a learner and train it1
    inSampleRMSE = [0]
    inSampleCorr = [0]
    outSampleRMSE = [0]
    outSampleCorr = [0]

    k=3
    n=k
    kRange = range(3,n+1)
    for k in kRange:

        learner = knn.KNNLearner(k) # constructor

        learner.addEvidence(trainX, trainY) # train it

        # evaluate in sample
        predY = learner.query(trainX) # get the predictions
        rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
        print
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

    # print 'Summary results:'
    #
    # for k in kRange:
    #     print 'k=', k, ' inRMSE=',inSampleRMSE[k], ' inCorr=',inSampleCorr[k],  ' outRMSE=',outSampleRMSE[k], ' outCorr=', outSampleCorr[k]
     
    # k=n
    # print 'k=', 3, ' inRMSE=',inSampleRMSE[k], ' inCorr=',inSampleCorr[k],  ' outRMSE=',outSampleRMSE[k], ' outCorr=', outSampleCorr[k]

#learner.addEvidence(Xtrain, Ytrain) # training step

    #plot knn data

    # # Plot RMSE vs k
    #
    # k = kRange
    # # print 'len k ', len(k)
    # y1 = inSampleRMSE[1:]
    # # print 'len y1 ',len(y1)
    # y2 = outSampleRMSE[1:]
    #
    # plt.plot(k, y1, 'ro-', label = 'In Sample RMSE')
    # plt.plot(k, y2, 'go-', label ='Out of Sample RMSE')
    # # plt.plot(k, y1,  'r', label ='in sample RMSE', k, y2, 'g',label='out sample RMSE')
    # plt.xlabel('k')
    # plt.ylabel('RMSE')
    # plt.title('RMSE vs k KNN with ripple data')
    # plt.show()
    #
    # #plot Corr vs k
    #
    # y1 = inSampleCorr[1:]
    # y2 = outSampleCorr[1:]
    #
    # plt.plot(k, y1, 'ro-', label ='In Sample Correlation')
    # plt.plot(k, y2, 'go-', label ='Out of Sample Correlation')
    # # plt.plot(k, y1,  'r', label ='in sample Corr', k, y2, 'g',label='out sample Corr')
    # plt.xlabel('k')
    # plt.ylabel('Corr')
    # plt.title('Correlation vs k KNN with ripple data')
    #
    # plt.show()

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


    # # for simple.csv
    # ax.set_xlim3d(0,11)
    # ax.set_ylim3d(0,11)
    # ax.set_zlim3d(0,11)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    plt.title('KNN, k = 3 on best4linreg data')
    plt.show()