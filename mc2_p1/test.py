"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import os
import csv

from util import get_data, plot_data
from portfolio.analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data

def test(orders_file):

  # read csv file:

    # get date range

    datesRead = pd.read_csv("orders/" + orders_file,

                        usecols=['Date'], # get rid of other columns and only leave those you want
                        )
    minDate = datesRead['Date'].min()
    maxDate = datesRead['Date'].max()

  # get date range
    dateRange = pd.date_range(minDate,maxDate)
    print dateRange

    # get symbols

    symbols = pd.read_csv("orders/" + orders_file,

                        usecols=['Symbol'], # get rid of other columns and only leave those you want
                        )

    print symbols['Symbol'][0]

    symbolReader = csv.reader(open("orders/" + orders_file, 'rU'))
    print 'symbolReader'
    symbolsList = []
    for col in symbolReader:
        if col[1] != 'Symbol':
            symbolsList.append(col[1])

    print symbolsList



    # read adjusted closing prices into the df


      #    if you use pandas to read the orders file, it will automatically sort it with a DatetimeIndex.

    ordersDf = pd.read_csv("orders/" + orders_file, index_col="Date",

                         # get rid of other columns and only leave those you want
                        )

    print 'ordersDf:', ordersDf

    # adjClosePrice = get_data(symbols, dates)


    # add a column, cash = 1, to prices file


    # return the value for each day = cash + the current value of equities.

test('orders-short.csv')