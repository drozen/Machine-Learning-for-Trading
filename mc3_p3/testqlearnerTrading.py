"""
Test a Q Learner in a navigation problem.  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand
import time
import math
import QLearner as ql
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import util
import marketsim


# move the robot according to the action and the map
def updateState(holding,a):

    # update holdings based on action
    if a == -1 and holding > 0: #sell
        holding -= 1
    elif a == 1: #buy
        holding +=1

    return holding

def discretizeThreshold(data, numSteps):
    '''
    :param data:
    :param numSteps:
    :return: thresholds used to discretize
    '''
    threshold = []
    stepsize = len(data) / numSteps
    data.sort()
    for i in range(0, numSteps):
        threshold.append(data[(i+1)*stepsize])

    return threshold

def discretize(value, threshold):
    for i in range(len(threshold)):
        if value > threshold[i] and value < threshold[i+1]:
            return i

# run the code to test a learner
if __name__=="__main__":

    #TODO: Verbose
    verbose = False #print lots of debug stuff if True

    # read in stock data

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

    # TODO: cumulative return
    #cum_ret = (port_val[-1]/port_val[0]) - 1

    # rename columns
    bb_value = bb_value.rename(columns ={filename:'bb_value'})
    volatility = volatility.rename(columns ={filename:'volatility'})
    momentum = momentum.rename(columns ={filename:'momentum'})

    # create empty dataframe with dates as index.

    plotDF = pd.DataFrame(index = dates)

    # join data frames

    plotDF = plotDF.join(bb_value, how='inner')

    stats = [volatility, momentum]

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

    trainX = trainData.iloc[:,1:4]
    trainY = trainData.iloc[:,-1]

    testX = testData.iloc[:,1:4]
    testY = testData.iloc[:,-1]

    #if verbose: printmap(data)

    rand.seed(5)

    # 10 states for 3 indicators
    learner = ql.QLearner(num_states=1000,\
        num_actions = 3, \
        rar = 0.98, \
        radr = 0.999, \
        verbose=verbose) #initialize the learner

    # # For Dyna
    #
    # learner = ql.QLearner(num_states=3,\
    #     num_actions = 3, \
    #     alpha = 0.2, \
    #     gamma = 0.9, \
    #     rar = 0.5, \
    #     radr = 0.99, \
    #     dyna = 100, \
    #     verbose=False) #initialize the learner


    # TODO: for training also make calls to query()
    for i in range(len(trainX)):
        steps = 0
        data = trainX
        print 'state:\n', state
        action = learner.querysetstate(state) #set the state and get first action

        #move to new location according to action and then get a new action
        newState = updateState(state,action)
            # rewards are daily return for each day of holding a position
        r = daily_returns[i]


        state = discretize(newState)
        action = learner.query(state,r)
        # TODO: remove sleep to speed up
        #if verbose: time.sleep(1)
        print i

    # TODO: for testing only make calls to querysetstate()