import csv
import pandas as pd
from util import get_data, plot_data

def test(orders_file):

    # read symbols
    symbolReader = csv.reader(open("orders/" + orders_file, 'rU'))
    symbols = []
    for col in symbolReader:
        if col[1] != 'Symbol':
            symbols.append(col[1])
    # make list unique
    symbols = list(set(symbols))

    print 'symbols:' , symbols

    dateReader = csv.reader(open("orders/" + orders_file, 'rU'))

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

    datesRead = pd.read_csv("orders/" + orders_file,

                        usecols=['Date'], # get rid of other columns and only leave those you want
                        )
    stDate = datesRead['Date'].min()
    endDate = datesRead['Date'].max()
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

    print 'CASH ADDED'
    print priceDf

        # create df
    tradesDf= pd.DataFrame(data=0.0, index=dateRange, columns=symbols)
     # drop na
    tradesDf = tradesDf.dropna()
    # add CASH column
    tradesDf['CASH'] = 0.0
    print 'tradesDf:'
    print tradesDf

    # read in orders file
    ordersDf = pd.read_csv("orders/" + orders_file, index_col='Date')
    print 'ordersDf'
    print ordersDf

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
        if order == 'BUY':
            tradesDf.ix[date, symbol] += shares
            tradesDf.ix[date,'CASH'] -= shares * priceDf.ix[date, symbol]

        elif order == 'SELL':
            tradesDf.ix[date, symbol] -= shares
            tradesDf.ix[date, 'CASH'] += shares * priceDf.ix[date, symbol]

    print'updated Trades Matrix'
    print tradesDf

    # "I got a work around from using for-loop by creating a new dataframe. The new dataframe is a simple multiplication of trade and price (with selected symbols) dataframe without cash column, and then row-wise sum of that new dataframe to get the required cash column."

    # holdings:
    # TODO: need to initialize the 1st cash with initial value
    start_val = 1000000.0
    holdingsDf = tradesDf
    holdingsDf.ix[stDate, 'CASH'] += start_val
    holdingsDf = holdingsDf.cumsum() # cumulutive sum to extend the values
    print 'holdingsDf'
    print holdingsDf

    valueDf = holdingsDf.multiply(priceDf)
    valueDf = valueDf.dropna()
    print 'valueDf'
    print valueDf

    portValDf = valueDf.sum(axis=1)
    print 'portValDf'
    print portValDf

test('orders-short.csv')