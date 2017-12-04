__author__ = 'Daniel Rozen'

import os
import pandas as pd #rename to pd for easy future reference
import numpy as np
import matplotlib.pyplot as plt

def test_run():
    #df = pd.read_csv("data/AAPL.csv")
    # print df # print entire data frame
    # print df.head() # print just the 1st 5 lines of csv
    #print df[10:21] # rows b/w index 10 and 20 - note 21 is non-inclusive

    # Define date range
    start_date='2010-01-01'
    end_date = '2010-12-31' # year 2010
    dates = pd.date_range(start_date, end_date)

    #Choose stock symbols to read
    symbols = ['GOOG', 'IBM', 'GLD'] # SPY will be added in get_data()

    # Get stock data
    df = get_data(symbols, dates)

    # # Slice by row range (dates_ using DataFrame.ix[] selector
    # print df.ix['2010-01-01':'2010-01-31'] # month of January - put in chronological order
    #
    # # Slice by column (symbols)
    # print df['GOOG'] # a signle label selects a single column
    # print df[['IBM', 'GLD']] # a list of labels selects multiple columns

    # slice by row and column
    print df.ix['2010-03-10':'2010-03-15', ['SPY', 'IBM']]

# utility functions

def symbol_to_path(symbol, base_dir= "C:/Users/danie/Documents/GitHub/ML4T/data/"):
    """Return CSV file path given ticker symbol."""
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols: # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        # TODO: Read and join data for each symbol

        df_temp=pd.read_csv(symbol_to_path(symbol), index_col='Date',
                            parse_dates=True, usecols=['Date','Adj Close'],na_values=['nan'])

        # Rename 'Adj Close' column to 'SPY' to prevent clash
        df_temp = df_temp.rename(columns={'Adj Close':symbol})

        df=df.join(df_temp) # use default how = 'left'
        if symbol == 'SPY': # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

if __name__ == "__main__":
    test_run()

