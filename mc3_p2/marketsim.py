"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import os
import csv

from util import get_data, plot_data
from portfolio.analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data

def compute_portvals(start_date, end_date, orders_file, start_val):
    """Compute daily portfolio value given a sequence of orders in a CSV file.

    Parameters
    ----------
        start_date: first date to track
        end_date: last date to track
        orders_file: CSV file to read orders from
        start_val: total starting cash available

    Returns
    -------
        portvals: portfolio value for each trading day from start_date to end_date (inclusive)
    """
    # TODO: Your code here


    # read symbols
    symbolReader = csv.reader(open(orders_file, 'rU'))
    symbols = []
    for col in symbolReader:
        if col[1] != 'Symbol':
            symbols.append(col[1])
    # make list unique
    symbols = list(set(symbols))

  # print 'symbols:' , symbols

    dateReader = csv.reader(open(orders_file, 'rU'))

    dateList = []
    for col in dateReader:
        if col[0] != 'Date':
            dateList.append(col[0])

    # make list unique
    dateList = list(set(dateList))

    # print 'dateList', dateList
     # sort
    # print 'sorted dateList', dateList.sort()
    # print 'dateList', dateList

    datesRead = pd.read_csv(orders_file,

                        usecols=['Date'], # get rid of other columns and only leave those you want
                        )

    # track st and end dates
    stDate = datesRead['Date'].min()

    endDate = datesRead['Date'].max()

    if start_date < stDate:
        stDate = start_date
    if end_date > endDate:
        endDate = end_date
    dateRangeList = [stDate, endDate]
    # print 'dateRangeList', dateRangeList

  # get date range
    dateRange = pd.date_range(stDate,endDate)
    # print dateRange

    # create df
    priceDf= get_data(symbols, dateRange, addSPY=False)

    # drop na
    priceDf = priceDf.dropna()

    # print 'dataframe:'
    # print priceDf

    # add Cash Df
    # cashDf = pd.DataFrame()
    # priceDf.add(cashDf, axis='columns', fill_value=0.0)

    priceDf['CASH'] = 1.0

  # print 'CASH ADDED'
  # print priceDf

        # create df
    tradesDf= pd.DataFrame(data=0.0, index=dateRange, columns=symbols)
     # drop na
    tradesDf = tradesDf.dropna()
    # add CASH column
    tradesDf['CASH'] = 0.0
  # print 'tradesDf:'
  # print tradesDf

    # read in orders file
    ordersDf = pd.read_csv(orders_file, index_col='Date')
  # print 'ordersDf'
  # print ordersDf

    # Iterate the orders file and fill the number of
    #shares for that particular symbol and date.

    # for date, symbol, order, shares in zip(ordersDf[0],ordersDf['Symbol'],ordersDf['Order'],ordersDf['Shares']):
    for index, row in ordersDf.iterrows():
	#index is your date
        date = index
        symbol = row['Symbol']
        order = row['Order']
        shares = row['Shares']
        # print 'date, symbol, order, shares:'
        # print date, symbol, order, shares
        if date >= start_date and date <= end_date:
            if order == 'BUY':
                tradesDf.ix[date, symbol] += shares
                tradesDf.ix[date,'CASH'] -= shares * priceDf.ix[date, symbol]

            elif order == 'SELL':
                tradesDf.ix[date, symbol] -= shares
                tradesDf.ix[date, 'CASH'] += shares * priceDf.ix[date, symbol]

  # print'updated Trades Matrix'
  # print tradesDf

    # "I got a work around from using for-loop by creating a new dataframe. The new dataframe is a simple multiplication of trade and price (with selected symbols) dataframe without cash column, and then row-wise sum of that new dataframe to get the required cash column."

    # holdings:
    # TODO: need to initialize the 1st cash with initial value
    holdingsDf = tradesDf
    holdingsDf.ix[stDate, 'CASH'] += start_val
    holdingsDf = holdingsDf.cumsum() # cumulutive sum to extend the values
  # print 'holdingsDf'
  # print holdingsDf

    valueDf = holdingsDf.multiply(priceDf)
    valueDf = valueDf.dropna()
  # print 'valueDf'
  # print valueDf

    portValDf = valueDf.sum(axis=1)
  # print 'portValDf'
  # print portValDf
    # return the value for each day = cash + the current value of equities.

    portvals = portValDf

    return portvals

def percentDiff(b,a):
    return (b-a)/(a+b)/2 * 100

def test_run():
    """Driver function."""
    # Define input parameters
    # start_date = '2011-01-05'
    # end_date = '2011-01-20'
    # orders_file = os.path.join("orders", "orders-short.csv")
    # start_val = 1000000

    # # Test 2:
    # start_date = '2011-01-10'
    # end_date = '2011-12-20'
    # orders_file = os.path.join("orders", "orders.csv")
    # start_val = 1000000

    # Test 3
    #TODO: change dates

    # # in sample
    # start_date = '2007-12-31'
    # end_date = '2009-12-31'

    # out of sample
    start_date = '2009-12-31'
    end_date = '2010-12-31'
    #orders_file = os.path.join("orders", "ordersMC2P2.csv")
    orders_file = os.path.join("orders.csv")

    start_val = 10000

    # Process orders
    portvals = compute_portvals(start_date, end_date, orders_file, start_val)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series
    
    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    # Simulate a $SPX-only reference portfolio to get stats
    prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['$SPX']]  # remove SPY
    portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    # Compare portfolio against $SPX
    print "Data Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of $SPX: {}".format(sharpe_ratio_SPX)
    print "Percent diff: ", percentDiff(sharpe_ratio, sharpe_ratio_SPX)
    print "\nCumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of $SPX: {}".format(cum_ret_SPX)
    print "Percent diff: ", percentDiff(cum_ret, cum_ret_SPX)

    print "\nStandard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of $SPX: {}".format(std_daily_ret_SPX)
    print "Percent diff: ", percentDiff(std_daily_ret, std_daily_ret_SPX)

    print "\nAverage Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of $SPX: {}".format(avg_daily_ret_SPX)
    print "Percent diff: ", percentDiff(avg_daily_ret, avg_daily_ret_SPX)

    print "Final Portfolio Value: {}".format(portvals[-1])

    # Plot computed daily portfolio value
    df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
    plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")


if __name__ == "__main__":
    test_run()
