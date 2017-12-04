__author__ = 'Daniel Rozen'
'''This code should generate a .png chart that illustrates the bollinger bands, and entry and exit points for a bollinger-based strategy.
It should also generate an orders.txt file that you feed into your market simulator to backtest the strategy
'''

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

    # generate a .png chart that illustrates the bollinger bands, and entry and exit points for a bollinger-based strategy.

    # Chart includes:

    # Stock price (adjusted close) over time
    # 20 day simple moving average (SMA) line

    SMA = pd.rolling_mean(price, window=20, min_periods = 20)
    # # normalize SMA
    # SMA = price/SMA - 1

    std = pd.rolling_std(price, window=20, min_periods = 20)
    # Lower Band = SMA - 2 * 20 day standard deviation.
    BB_low = SMA - 2*std
    # Upper Band = SMA + 2 * 20 day standard deviation.
    BB_up = SMA + 2*std

    # Long entries as a vertical green line at the time of entry.
    # The long entry is made when the price transitions from below the lower band to above the lower band
    # given two arrays, a and b
    a= price['IBM']['2008-01-29':]
    b= BB_low['IBM']['2008-01-29':]
    # every 1 means A crossover from A being less than B to A being greater than B
    longEntries = np.pad(np.diff(np.array(a > b).astype(int)),
       (1,0), 'constant', constant_values = (0,))

    # print 'longEntries: ', longEntries
    longEntriesIndex = np.where(longEntries==1)
    # print 'longEntriesIndex: ', longEntriesIndex

    # Long exits as a vertical black line at the time of exit.
    # The exit signal occurs when the price moves from below the SMA to above it.
    a= price['IBM']['2008-01-29':]
    b= SMA['IBM']['2008-01-29':]
    # every 1 means A crossover from A being less than B to A being greater than B and every -1 means A crossover from A being greater than B to A being less than B
    longExits = np.pad(np.diff(np.array(a > b).astype(int)),
       (1,0), 'constant', constant_values = (0,))
    longExitsIndex = np.where(longExits==1)

    # Short entries as a vertical RED line at the time of entry
    # short entry is made when the price transitions from above the upper band to below the upper band
    a= price['IBM']['2008-01-29':]
    b= BB_up['IBM']['2008-01-29':]
    #print 'BB_up[\'IBM\']:', BB_up['IBM']
    # every 1 means A crossover from A being less than B to A being greater than B and every -1 means A crossover from A being greater than B to A being less than B
    shortEntries = np.pad(np.diff(np.array(a > b).astype(int)),
       (1,0), 'constant', constant_values = (0,))

    #print 'shortEntries: ', shortEntries
    shortEntriesIndex = np.where(shortEntries==-1)
    print '\nshortEntriesIndex: \n', shortEntriesIndex

    # Short exits as a vertical black line at the time of exit.
    # exit signal occurs when the price moves from above the SMA to below it.

    a= price['IBM']['2008-01-29':]
    b= SMA['IBM']['2008-01-29':]
    #print 'BB_up[\'IBM\']:', BB_up['IBM']
    # every 1 means A crossover from A being less than B to A being greater than B and every -1 means A crossover from A being greater than B to A being less than B
    shortExits = np.pad(np.diff(np.array(a > b).astype(int)),
       (1,0), 'constant', constant_values = (0,))

    #print 'shortExits: ', shortExits
    shortExitsIndex = np.where(shortExits==-1)
    #print 'shortExitsIndex: ', shortExitsIndex

    # create empty dataframe with dates as index.

    plotDF = pd.DataFrame(index = dates)

    # join data frames

    plotDF = plotDF.join(price, how='inner')

    # rename columns
    SMA = SMA.rename(columns ={'IBM':'SMA'})
    BB_up = BB_up.rename(columns ={'IBM':' BB_upper'})
    BB_low = BB_low.rename(columns ={'IBM':' BB_lower'})

    stats = [SMA, BB_up, BB_low]
    for stat in stats:
        plotDF = plotDF.join(stat)
        
    # add long entries/exits to dataframe
    plotDFEntries = plotDF['2008-01-29':]

    # indexes = [longEntriesIndex, longExitsIndex, shortEntriesIndex, shortExitsIndex]
    # for index in indexes:
    #     plotDFEntries = plotDFEntries.join(index)
    plotDFEntries.loc[:,('Long Entries')] = longEntries
    plotDFEntries.loc[:,('Long Exits')] = longExits
    plotDFEntries.loc[:,('Short Entries')] = shortEntries
    plotDFEntries.loc[:,('Short Exits')] = shortExits

    print '\n plotDFEntries: \n', plotDFEntries[['Long Entries', 'Long Exits', 'Short Entries', 'Short Exits']]['2008-09-25':'2008-12-10']

    # filter out entries/exits to match corresponding ones

    dfLong = plotDFEntries[(plotDFEntries['Long Entries']== 1) | (plotDFEntries['Long Exits']== 1) | (plotDFEntries['Short Entries']== -1) | (plotDFEntries['Short Exits']== -1)]

    print '\n dfLong before consecutive entries exits filters: \n', dfLong[['Long Entries', 'Long Exits', 'Short Entries', 'Short Exits']]
    # TODO: edit filters to find out issues

    # remove consecutive entries/exits
    dfLong = dfLong[((dfLong['Long Entries'].shift(1) == 0) & (dfLong['Long Entries'] == 1)) | ((dfLong['Long Exits'].shift(1) == 0) & (dfLong['Long Exits'] == 1)) | ((dfLong['Short Entries'].shift(1) == 0) & (dfLong['Short Entries'] == -1)) | ((dfLong['Short Exits'].shift(1) == 0) & (dfLong['Short Exits'] == -1)) | ((dfLong['Long Entries'].shift(1) == -1) & (dfLong['Long Entries'] == 1)) | ((dfLong['Long Exits'].shift(1) == -1) & (dfLong['Long Exits'] == 1)) | ((dfLong['Short Entries'].shift(1) == 1) & (dfLong['Short Entries'] == -1)) | ((dfLong['Short Exits'].shift(1) == 1) & (dfLong['Short Exits'] == -1))]

    # # # negation style
    # dfLong = dfLong[~((dfLong['Long Entries'].shift(1) == 1) & (dfLong['Long Entries'] == 1)) | ((dfLong['Long Exits'].shift(1) == 1) & (dfLong['Long Exits'] == 1)) | ((dfLong['Short Entries'].shift(1) == -1) & (dfLong['Short Entries'] == -1)) | ((dfLong['Short Exits'].shift(1) == -1) & (dfLong['Short Exits'] == -1))]
    # only retain an entry with an exit directly after it

    dfComparePrev = dfLong[((dfLong['Long Entries'].shift(1) == 1) & (dfLong['Long Exits'] == 1)) | ((dfLong['Long Exits'].shift(-1) == 1) & (dfLong['Long Entries'] == 1)) | ((dfLong['Short Entries'].shift(1) == -1) & (dfLong['Short Exits'] == -1)) | ((dfLong['Short Exits'].shift(-1) == -1) & (dfLong['Short Entries'] == -1))]

    print '\ndfComparePrev: \n', dfComparePrev[['Long Entries', 'Long Exits', 'Short Entries', 'Short Exits']]

    # combine long entries and exits
    # When this entry signal criteria is met, buy the stock and hold it until the exit.
    csvOutput = ['Date,Symbol,Order,Shares']

    # date files for graphing
    shortEntriesDates = []
    shortExitsDates = []
    longEntriesDates = []
    longExitsDates = []

    for index, row in dfComparePrev.iterrows():
        if row['Short Entries'] == -1:
            string = str(index.date()) + ",IBM,SELL,100"
            csvOutput.append(string)
            shortEntriesDates.append(str(index.date()))
        if row['Short Exits'] == -1:
            string = str(index.date()) + ",IBM,BUY,100"
            csvOutput.append(string)
            shortExitsDates.append(str(index.date()))

        if row['Long Entries'] == 1:
            string = str(index.date()) + ",IBM,BUY,100"
            csvOutput.append(string)
            longEntriesDates.append(str(index.date()))

        if row['Long Exits'] == 1:
            string = str(index.date()) + ",IBM,SELL,100"
            csvOutput.append(string)
            longExitsDates.append(str(index.date()))

    print 'csvOutput: \n', csvOutput

    # convert ouptut to txt file
    b = open('orders.csv', 'w')
    for out in csvOutput:
        b.write(out)
        b.write('\n')
    b.close()

    print '\nlongEntriesDates: \n', longEntriesDates
    print '\nlongExitsDates: \n ', longExitsDates
    
    #shortEntriesDates = plotDF['2008-01-29':].index[shortEntriesIndex]
    #shortExitsDates = plotDF['2008-01-29':].index[shortExitsIndex]
    print '\nshortEntriesDates: \n', shortEntriesDates
    print '\nshortExitsDates:  \n', shortExitsDates
    
  # print '\nplotDFEntries: \n', plotDFEntries

    # Plot data:
    ax = plotDF.plot(title="Stock prices", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    # plot vertical lines
    plt.vlines(x = longEntriesDates, ymin = 60, ymax = 130, colors = 'green')
    plt.vlines(x = longExitsDates, ymin = 60, ymax = 130, colors = 'black')
    plt.vlines(x = shortEntriesDates, ymin = 60, ymax = 130, colors = 'red')
    plt.vlines(x = shortExitsDates, ymin = 60, ymax = 130, colors = 'black')

    plt.show()

    # plot vertical lines

if __name__ == "__main__":
    test_run()

