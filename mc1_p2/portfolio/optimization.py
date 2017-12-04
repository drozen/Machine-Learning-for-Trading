"""MC1-P2: Optimize a portfolio."""
import numpy as np

import pandas as pd

import scipy.optimize as spo

from util import get_data, plot_data
from analysis import get_portfolio_value, get_portfolio_stats
# use this if the above import doesn't work
# from portfolio.analysis import get_portfolio_value, get_portfolio_stats
allocs = np.array([.25,.25,.25,.25])
def f(allocs, prices):

    port_val = get_portfolio_value(prices, allocs)
    # Get portfolio statistics (note: std_daily_ret = volatility)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(port_val)

    return -1 * sharpe_ratio

def find_optimal_allocations(prices):
    """Find optimal allocations for a stock portfolio, optimizing for Sharpe ratio.

    Parameters
    ----------
        prices: daily prices for each stock in portfolio

    Returns
    -------
        allocs: optimal allocations, as fractions that sum to 1.0
    """
    # TODO: Your code here

    # Get daily portfolio value (already normalized since we use default start_val=1.0)

    #
    #port_val = get_portfolio_value(prices, allocs)

    # # Get portfolio statistics (note: std_daily_ret = volatility)
    # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(port_val)

    # 1. provide function to minimize f(x) = Sharpe ratio * -1

    # 2. provide initial guess for x

    # Constraints for optimizer

        # output must be b/w 0 and 1

    bounds=tuple([(0,1)]*4)

    # use:   tuple([(0,1)]*10)

        # sum of allocations = 1

    # sum(abs(x_i)) = 1

    constraints = ({ 'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs) })

    # 3. call the optimizer
    x0 = np.array([.25,.25,.25,.25])

    # result = x0[0]**2 + x0[1]**2 + x0[2]**2 + x0[3]**2
    #print 'result: ', result
    OptimizeResult = spo.minimize(f, x0=x0, args=(prices,), method = 'SLSQP', bounds=bounds, constraints=constraints, options = {'disp': True})
    allocs = OptimizeResult.x
    return allocs


def optimize_portfolio(start_date, end_date, symbols):
    """Simulate and optimize portfolio allocations."""
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get optimal allocations
    allocs = find_optimal_allocations(prices)
    allocs = allocs / np.sum(allocs)  # normalize allocations, if they don't sum to 1.0

    # Get daily portfolio value (already normalized since we use default start_val=1.0)
    port_val = get_portfolio_value(prices, allocs)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(port_val)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Optimal allocations:", allocs
    print "Sharpe Ratio:", sharpe_ratio
    print "Volatility (stdev of daily returns):", std_daily_ret
    print "Average Daily Return:", avg_daily_ret
    print "Cumulative Return:", cum_ret

    # Compare daily portfolio value with normalized SPY
    normed_SPY = prices_SPY / prices_SPY.ix[0, :]
    df_temp = pd.concat([port_val, normed_SPY], keys=['Portfolio', 'SPY'], axis=1)
    plot_data(df_temp, title="Daily Portfolio Value and SPY")


def test_run():
    """Driver function."""
    # Define input parameters

    start_date = '2010-01-01'
    end_date = '2010-12-31'
    symbols = ['GOOG', 'AAPL', 'GLD', 'HNZ']  # list of symbols

    # test 1
    # start_date = '2010-01-01'
    # end_date = '2010-12-31'
    # symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']  # list of symbols

    # test 2
    #
    # start_date = '2004-01-01'
    # end_date = '2006-01-01'
    # symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']  # list of symbols
    #
    # # test 3
    # start_date = '2004-12-01'
    # end_date = '2006-05-31'
    # symbols = ['YHOO', 'XOM', 'GLD', 'HNZ']  # list of symbols

    #     # test 4
    # start_date = '2005-12-01'
    # end_date = '2006-05-31'
    # symbols = ['YHOO', 'HPQ', 'GLD', 'HNZ']  # list of symbols

    # Optimize portfolio
    optimize_portfolio(start_date, end_date, symbols)


if __name__ == "__main__":
    test_run()
