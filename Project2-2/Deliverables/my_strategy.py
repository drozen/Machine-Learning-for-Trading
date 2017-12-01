__author__ = 'Daniel Rozen'
'''This code should generate a .png chart (or charts) that illustrate the clever strategy you develop, along with indications of entry and exit points.
It should also generate an orders.txt file that you feed into your market simulator to backtest the strategy.'''

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# set pandas display options
# widen print output
pd.set_option('display.max_columns', 100)

# Utility functions

def symbol_to_path(symbol, base_dir=os.path.join("..", "data")):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


def get_data(symbols, dates, addSPY=True):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df


def plot_data(df, tradeDates, title="Stock prices", xlabel="Date", ylabel="Price"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.vlines(x = tradeDates, ymin = 60, ymax = 130, colors = 'green')

    plt.show()

def plot_selected(df, columns, start_index, end_index):
    """ Plot the desired columns over the index values in the given range."""
    plot_data(df.ix[start_index:end_index,columns],title="selected data")


# My functions

def normalize(data):
    return (data - data.mean())/data.std()

def daily_change(df):
        # Rather do it much easier with Pandas!
    daily_change = (df/ df.shift(1)) - 1
    # Note: Returned DataFrame must have the same number of rows
    daily_change.ix[0,:] = 0 # set daily returns for row 0 to 0 because they can't be computed
    return daily_change


def test_run():

    # widen print output
    pd.set_option('display.max_columns', 100)

    """Driver function."""
    # Define input parameters
    start_date = '2007-12-31'

    end_date = '2009-12-31'
    dates = pd.date_range(start_date, end_date)
    symbols = ['IBM']

    price = get_data(symbols, dates, addSPY=False)
    price = price.dropna()
    priceNorm = (price / price['IBM'][0]) * 20 -20

    # 20 day simple moving average (SMA) line

    SMA = pd.rolling_mean(price, window=20, min_periods = 20)

    std = pd.rolling_std(price, window=20, min_periods = 20)
    # Lower Band = SMA - 2 * 20 day standard deviation.
    BB_low = SMA - 2*std
    # Upper Band = SMA + 2 * 20 day standard deviation.
    BB_up = SMA + 2*std

    # GET SPY DATA:
    symbols = ['SPY']

    priceSPY = get_data(symbols, dates, addSPY=False)
    priceSPY = priceSPY.dropna()

    # 20 day simple moving average (SMA) line

    SMASPY = pd.rolling_mean(priceSPY, window=20, min_periods = 20)

    stdSPY = pd.rolling_std(priceSPY, window=20, min_periods = 20)
    # Lower Band = SMA - 2 * 20 day standard deviation.
    BB_lowSPY = SMASPY - 2*stdSPY
    # Upper Band = SMA + 2 * 20 day standard deviation.
    BB_upSPY = SMASPY + 2*stdSPY

    # bollingerValueSPY = (priceSPY - BB_lowSPY) / (BB_upSPY - BB_lowSPY)
    # bollingerValue = (price - BB_low) / (BB_up - BB_low)

    # Capt Balch's version

    # IBM bollinger value
    bollingerValue = 2*(price - SMA) / (BB_up - BB_low)
    bollingerValueSPY = 2*(priceSPY - SMASPY) / (BB_upSPY - BB_lowSPY)

    # calculate daily changes of Bollinger values:
    DCBol = daily_change(bollingerValue)
    DCBolSPY = daily_change(bollingerValueSPY)

    # ratio
    bollingerRatio = bollingerValue/bollingerValueSPY.squeeze()

    print '\nbollingerRatio: \n', bollingerRatio

    # ENTRIES AND EXITS:

    # Long entries as a vertical green line at the time of entry.
    # The long entry is made when the price transitions from below the lower band to above the lower band
    # given two arrays, a and b

    # create empty dataframe with dates as index.

    plotDF = pd.DataFrame(index = dates)

    # join data frames

    plotDF = plotDF.join(price, how='inner')

    # rename columns
    SMA = SMA.rename(columns ={'IBM':'SMA'})
    BB_up = BB_up.rename(columns ={'IBM':'BB_upper'})
    BB_low = BB_low.rename(columns ={'IBM':'BB_lower'})

    BB_upSPY = BB_upSPY.rename(columns ={'SPY':'BB_upper_SPY'})
    BB_lowSPY = BB_lowSPY.rename(columns ={'SPY':'BB_lower_SPY'})

    bollingerValueSPY = bollingerValueSPY.rename(columns ={'SPY':'bollingerValueSPY'})

    bollingerValue = bollingerValue.rename(columns ={'IBM':'bollingerValue'})

    bollingerRatio = bollingerRatio.rename(columns ={'IBM':'bollingerRatio'})

    priceNorm = priceNorm.rename(columns ={'IBM':'IBM_normalized'})

    DCBol = DCBol.rename(columns ={'IBM':'Daily_Change_Bollinger_Value'})
    DCBolSPY = DCBolSPY.rename(columns ={'SPY':'Daily_Change_Bollinger_Value_SPY'})

    # stats = [SMA, BB_up, BB_low, BB_upSPY, BB_lowSPY]
    #stats = [priceNorm, bollingerRatio, bollingerValue, bollingerValueSPY]
    stats = [priceNorm, DCBol, DCBolSPY]

    for stat in stats:
        plotDF = plotDF.join(stat)

    # add long entries/exits to dataframe
    plotDFEntries = plotDF['2008-01-29':]

    # TODO: adjust buyThreshold
    buyThreshold = 10
    dfBuySell = plotDFEntries[((plotDFEntries['Daily_Change_Bollinger_Value'] > buyThreshold) | (plotDFEntries['Daily_Change_Bollinger_Value'] < buyThreshold * -1))]
    # dfBuySell = plotDFEntries[(((plotDFEntries['Daily_Change_Bollinger_Value'] > buyThreshold) & plotDFEntries['Daily_Change_Bollinger_Value_SPY'] < 0) | ((plotDFEntries['Daily_Change_Bollinger_Value'] < buyThreshold * -1) & plotDFEntries['Daily_Change_Bollinger_Value_SPY'] > 0))]


    print "dfBuySell: \n", dfBuySell

    csvOutput = ['Date,Symbol,Order,Shares']

    # date files for graphing
    longEntriesDates = []
    longExitsDates = []

    prevRow = 0
    for index, row in dfBuySell.iterrows():
        if prevRow == 0:
            if row['Daily_Change_Bollinger_Value'] > buyThreshold:
                string = str(index.date()) + ",IBM,BUY,100"
                csvOutput.append(string)
                prevRow = row['Daily_Change_Bollinger_Value']
                longEntriesDates.append(str(index.date()))
            if row['Daily_Change_Bollinger_Value'] < buyThreshold * -1:
                string = str(index.date()) + ",IBM,SELL,100"
                csvOutput.append(string)
                prevRow = row['Daily_Change_Bollinger_Value']
                longExitsDates.append(str(index.date()))
        else:
            if row['Daily_Change_Bollinger_Value'] > buyThreshold and prevRow < buyThreshold* -1:
                string = str(index.date()) + ",IBM,BUY,100"
                csvOutput.append(string)
                prevRow = row['Daily_Change_Bollinger_Value']
                longEntriesDates.append(str(index.date()))
            if row['Daily_Change_Bollinger_Value'] < buyThreshold * -1 and prevRow > buyThreshold:
                string = str(index.date()) + ",IBM,SELL,100"
                csvOutput.append(string)
                prevRow = row['Daily_Change_Bollinger_Value']
                longExitsDates.append(str(index.date()))

    print 'csvOutput: \n', csvOutput

    # convert ouptut to txt file
    b = open('orders.csv', 'w')
    for out in csvOutput:
        b.write(out)
        b.write('\n')
    b.close()


  # print '\nplotDFEntries: \n', plotDFEntries

    # print '\nplotDf[[ , ]]\n', plotDF
    # Plot data:
    #ax = plotDF[['IBM_normalized','bollingerValue','bollingerValueSPY']].plot(title="Stock prices", fontsize=12)
    ax = plotDF[['IBM_normalized', 'Daily_Change_Bollinger_Value']].plot(title="Stock prices", fontsize=12)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    # plot vertical lines
    plt.vlines(x = longEntriesDates, ymin = -15, ymax = 15, colors = 'red')
    plt.vlines(x = longExitsDates, ymin = -15, ymax = 15, colors = 'black')
    # plt.vlines(x = shortEntriesDates, ymin = 60, ymax = 130, colors = 'red')
    # plt.vlines(x = shortExitsDates, ymin = 60, ymax = 130, colors = 'black')

    plt.show()

    # plot vertical lines

if __name__ == "__main__":
    test_run()

