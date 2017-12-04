__author__ = 'Daniel Rozen'
'''transform your regression learner into a stock trading strategy
'''
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import KNNLearner as knn
import LinRegLearner as lrl
import BagLearner as bl
import math
import pandas as pd
import util

# set pandas display options
# widen print output
#pd.set_option('display.max_columns', 100)

def test_run():

    # widen print output
    pd.set_option('display.max_columns', 100)

    """Driver function."""
    # Define input parameters
    start_date = '2007-12-31'
    end_date = '2010-12-31'
    dates = pd.date_range(start_date, end_date)
    # TODO: choose filename stock
    filename = 'ML4T-399'
    #filename = 'IBM'
    symbols = [filename]
    price = util.get_data(symbols, dates, addSPY=False)
    price = price.dropna()

    # # normalize price
    normedPrice = price / price[filename][0]

    # calculate stats:
    SMA = pd.rolling_mean(price, window=20, min_periods = 20)
    # volatility
    std = pd.rolling_std(price, window=20, min_periods = 20)
    BB_low = SMA - 2*std
    BB_up = SMA + 2*std
    bollingerValue = 2*(price - SMA) / (BB_up - BB_low)

    # calculate x:
    # bb_value[t] = (price[t] - SMA[t])/(2 * stdev[t])
    bb_value = (price - SMA) /(2*std)
    #momentum[t] = (price[t]/price[t-N]) - 1   N = # days
    N = 10
    momentum = price/price.shift(N) - 1

    # daily returns
    daily_returns = (price/ price.shift(1)) - 1

    # volatilty[t] = stddev(dailyreturns)
    volatility = pd.rolling_std(daily_returns, window=20, min_periods = 20)

    # calculate Y[t] = (price[t+5]/price[t]) - 1.0
    FiveDayReturn = price.shift(-5)/price - 1
    FiveDayReturnPrice = price.shift(-5)

    # create empty dataframe with dates as index.

    plotDF = pd.DataFrame(index = dates)

    # join data frames

    plotDF = plotDF.join(price, how='inner')

    # rename columns
    bb_value = bb_value.rename(columns ={filename:'bb_value'})
    volatility = volatility.rename(columns ={filename:'volatility'})
    momentum = momentum.rename(columns ={filename:'momentum'})

    FiveDayReturn = FiveDayReturn.rename(columns ={filename:'FiveDayReturn'})

    stats = [bb_value, volatility, momentum, FiveDayReturn]

    for stat in stats:
        plotDF = plotDF.join(stat)

  # print 'plotDF: \n', plotDF

    # Using KNN learner:

    # sort out train Data and test data
    trainData = plotDF.ix['2007-12-31':'2009-12-31']
    testData = plotDF.ix['2009-12-31':'2010-12-31']
    # drop NA values
    trainData = trainData.dropna()
    testData = testData.dropna()
    #   # convert to ND array
    # trainData = trainData.as_matrix()
    # testData = testData.as_matrix()

    # random samplings
    # trainIndex = np.random.choice(range(len(trainData)), size = len(trainData), replace = False)
    # testIndex = np.random.choice(range(len(testData)), size = len(testData), replace = False)
    # trainIndex = sorted(trainIndex)
    # testIndex = sorted(testIndex)

    trainX = trainData.iloc[:,1:4]
    trainY = trainData.iloc[:,-1]

    testX = testData.iloc[:,1:4]
    testY = testData.iloc[:,-1]

    # create a learner and train it
    inSampleRMSE = [0]
    inSampleCorr = [0]
    outSampleRMSE = [0]
    outSampleCorr = [0]

    k=5
    kwargs = {"k":k}
    # TODO: change to 20.  Only at 5 for quicker debug purposes
    bags = 20

    ##loop through k values
    # for k in range(3,10):
    # learner = bl.BagLearner(learner = knn.KNNLearner, kwargs = {"k":k}, bags = bags, boost = False)

    # TODO: CHOOSE LEARNER
    # #use KNN learner
    learner = bl.BagLearner(learner = knn.KNNLearner, kwargs = kwargs, bags = bags, boost = False)

    # #use LinRegLearner
    # learner = bl.BagLearner(learner = lrl.LinRegLearner, kwargs = {}, bags = bags, boost = False)
    #   without bags:
    #learner = lrl.LinRegLearner()

    #
    #   # convert to ND array
    trainXarray = trainX.as_matrix()
    trainYarray = trainY.as_matrix()

    testXarray = testX.as_matrix()
    testYarray = testY.as_matrix()

    priceArray = price.as_matrix()

    learner.addEvidence(trainXarray, trainYarray)

    # k=12
    # n=k
    # for k in range(1,n+1):
    #
    #     learner = knn.KNNLearner(k) # constructor
    #
    #     learner.addEvidence(trainXarray, trainYarray) # train it

    # evaluate in sample
    predYinSample = learner.query(trainXarray) # get the predictions
    rmse = math.sqrt(((trainYarray - predYinSample) ** 2).sum()/trainYarray.shape[0])
    print "Number of bags: ", bags
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predYinSample, y=trainYarray)
    print "corr: ", c[0,1]

    inSampleRMSE.append(rmse)
    inSampleCorr.append(c[0,1])

    # evaluate out of sample
    predYoutSample = learner.query(testXarray) # get the predictions
    rmse = math.sqrt(((testYarray - predYoutSample) ** 2).sum()/testYarray.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predYoutSample, y=testYarray)
    print "corr: ", c[0,1]

    outSampleRMSE.append(rmse)
    outSampleCorr.append(c[0,1])

       # 	Create a plot that illustrates your training Y values in one color, price in another color and your model's PREDICTED Y in a third color
    # predicted Y is predicted price
#    onedf = pd.DataFrame(data=1., index = trainY.index)
 #   trainYprice = price.mult(trainY.add(onedf))

#    print 'trainYprice:\n', trainYprice

    predYinSampleDF = pd.DataFrame(list(predYinSample), index = trainX.index,  )
    predYoutSampleDF = pd.DataFrame(list(predYoutSample), index = testX.index)

    predYtotalDF = predYinSampleDF.append(predYoutSampleDF)  # add train and test results together
    # predYinSample1Df = (1+predYinSampleDF.ix['2007-12-31':'2009-12-31'])
    # predYinSample1Df = predYinSample1Df.rename(columns ={0:filename})
    predYinSampleDF = predYinSampleDF.rename(columns ={0:filename})
    predYoutSampleDF = predYoutSampleDF.rename(columns ={0:filename})

    predYinSamplePrice = price.ix['2008-01-29':'2009-12-31'].multiply(1+predYinSampleDF)
    predYoutSamplePrice = price.ix['2009-12-31':'2010-12-23'].multiply(1+predYoutSampleDF)

  # print 'predYinSamplePrice\n', predYinSamplePrice

    plt.plot(FiveDayReturnPrice.index, FiveDayReturnPrice, 'g', label = 'training Y values')
    plt.plot(price.ix['2007-12-31':'2009-12-31'].index, price.ix['2007-12-31':'2009-12-31'], 'b', label ='price')
    # TODO: choose in sample / or out of sample price plot
    plt.plot(predYinSamplePrice.index, predYinSamplePrice, 'r', label ='PREDICTED Y')
    plt.xlabel('date')
    plt.ylabel('Price')
    plt.xlim('2008-03-01','2008-08-01')
    plt.title("Price / Training Y / Predicted Y")
    plt.legend()

    plt.show()
    #

    # plotDF1 = pd.DataFrame(index = dates)
    #   # rename columns
    # trainYprice = trainYprice.rename(columns ={filename:'trainYprice'})
    # predYprice = predYprice.rename(columns ={filename:'predYprice'})
    #
    # stats = [price, trainYprice, predYprice]
    #
    # for stat in stats:
    #     plotDF1 = plotDF1.join(stat)
    #
    predYtotalDF = predYtotalDF.rename(columns ={0:'pred5dayPercInc'})
    plotDF= plotDF.join(predYtotalDF)

    # drop NA
    plotDF = plotDF.dropna()

   # 	Create a trading policy based on what your learner predicts for future return.
    # As an example you might choose to buy when the forecaster predicts the price will go up more than 1%, then hold for 5 days.

    # add long entries/exits to dataframe


  #  print 'plotDFEntries:\n ' , plotDFEntries

    # TODO: adjust buyThreshold
    buyThreshold = .01
    indicator = 'pred5dayPercInc'
    # dfBuySell = plotDFEntries[((plotDFEntries[indicator] > buyThreshold))]
    dfBuySell = plotDF

  # print "dfBuySell: \n", dfBuySell

    csvOutput = ['Date,Symbol,Order,Shares']

    # date files for graphing
    longEntriesDates = []
    longExitsDates = []


    #TODO: change dates

    # # in sample
    # start_date = '2007-12-31'
    # end_date = '2009-12-31'

    # out of sample
    start_date = '2009-12-31'
    end_date = '2010-12-31'

    prevRow = 0
    for index, row in dfBuySell.iterrows():
        if str(index.date()) >= start_date and str(index.date()) <= end_date:
            if prevRow == 0:
                if row[indicator] > buyThreshold:
                    string = str(index.date()) + "," + str(symbols[0]) + ",BUY,100"
                    csvOutput.append(string)
                    prevRow = row[indicator]
                    longEntriesDates.append(str(index.date()))
                if row[indicator] < buyThreshold * -1:
                    string = str(index.date()) + "," + str(symbols[0]) + ",SELL,100"
                    csvOutput.append(string)
                    prevRow = row[indicator]
                    longExitsDates.append(str(index.date()))

            else:
                if row[indicator] > buyThreshold and prevRow < buyThreshold* -1:
                    string = str(index.date()) + "," + str(symbols[0]) + ",BUY,100"
                    csvOutput.append(string)
                    prevRow = row[indicator]
                    longEntriesDates.append(str(index.date()))
                if row[indicator] < buyThreshold * -1 and prevRow > buyThreshold:
                    string = str(index.date()) + "," + str(symbols[0]) + ",SELL,100"
                    csvOutput.append(string)
                    prevRow = row[indicator]
                    longExitsDates.append(str(index.date()))

  # print 'csvOutput: \n', csvOutput

    # convert ouptut to txt file
    b = open('orders.csv', 'w')
    for out in csvOutput:
        b.write(out)
        b.write('\n')
    b.close()

    #  	Create a plot that illustrates entry and exits as vertical lines on a price chart for the in sample period 2008-2009. Show long entries as green lines, short entries as red lines and exits as black lines
  # print '\nplotDFEntries: \n', plotDFEntries

    # print '\nplotDf[[ , ]]\n', plotDF
    # Plot data:
    #ax = plotDF[['IBM_normalized','bollingerValue','bollingerValueSPY']].plot(title="Stock prices", fontsize=12)
    ax = plotDF[[str(symbols[0])]].plot(title="Stock prices", fontsize=12)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    # TODO: change limits
    # in sample:
    # plt.xlim('2008-03-01','2008-08-01')
    # # # out of sample
    # plt.xlim('2010-03-01','2010-08-01')

    plt.xlim(start_date,end_date)

    #plot vertical lines
    ymin = 0
    ymax = 200
    plt.vlines(x = longEntriesDates, ymin = ymin, ymax = ymax, colors = 'green', label = 'long entries')
    plt.vlines(x = longExitsDates, ymin = ymin, ymax = ymax, colors = 'black', label = 'long exits')
    # plt.vlines(x = shortEntriesDates, ymin = 60, ymax = 130, colors = 'red')
    # plt.vlines(x = shortExitsDates, ymin = 60, ymax = 130, colors = 'black')
    plt.title("Entries and Exits vs. Price")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    test_run()
