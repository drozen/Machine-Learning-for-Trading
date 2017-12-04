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
    end_date = '2010-12-31'
    dates = pd.date_range(start_date, end_date)

    # Create an empty dataframe
    df1 = pd.DataFrame(index=dates)

    # Read SPY data into temporary dataframe .
    # Sort by dates with index_col = "Date",
    path1 = "C:/Users/danie/Documents/GitHub/ML4T/"
    dfSPY = pd.read_csv(path1 + "data/SPY.csv",
                        index_col = "Date", parse_dates=True,     # use parse_Dates = True to convert the dates present in the DataFrame into date time index objects

                        usecols=['Date','Adj Close'], # get rid of other columns and only leave those you want
                        na_values=['nan']) # interpret that NaN is not a number

    # Rename 'Adj Close' column to 'SPY' to prevent clash
    dfSPY = dfSPY.rename(columns={'Adj Close':'SPY'})

    # # Join the two dataframes using Datarame.join(), with how='inner'
    df1=df1.join(dfSPY, how='inner')

    # Read in more stocks into a combined data frame
    symbols = ['GOOG', 'IBM', 'GLD']
    # for loop to read and join each stock into the dataframe just like SPY
    for symbol in symbols:
        df_temp=pd.read_csv(path1+"data/{}.csv".format(symbol), index_col='Date',
                            parse_dates=True, usecols=['Date','Adj Close'],na_values=['nan'])

        df=df1.join(df_temp) # use default how = 'left'

    # for remaining rows, pandas introduces NaNs
    # # Drop all rows with NaN VAlues
    # df1 = df1.dropna()

    print df1

# utility functions

def symbol_to_path(symbol, base_dir= "C:/Users/Daniel Rozen/Anaconda/ml4t/data"):
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
        df_temp = df_temp.rename(columns={'Adj Close':'symbol'})


        df=df.join(df_temp) # use default how = 'left'
        if symbol == 'SPY': # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

if __name__ == "__main__":
    test_run()

