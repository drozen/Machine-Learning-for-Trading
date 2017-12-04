Readme how to run code from code.py

line 30-31 choose filename you desire: ML4T-399 / IBM

line 51 choose N momemtum window

line 119 - choose number of bags for bag learner

lines 125-132 - choose learner type: KNN / LinReg

line 237 - choose buy threshold for predicted 5day Percentage Increase

lines 253-259 - choose in / out of sample dates to conduct trades on

lines 308- 312 change plot limits



Step 1. run code.py,  this will output summary results eg.

Number of bags:  10
In sample results
RMSE:  0.0200616850273
corr:  0.991350132031


and then graph 1 will pop up

close graph 1 to see graph 2
close graph 2 to continue running the program to see out of sample results

Out of sample results
RMSE:  0.0100011981113
corr:  0.99775796885

The program will output orders.csv


Now open marketsim.py

lines 173-181  -  choose in sample or out of sample simulation dates

run marketsim.py

it will output summary data:

Data Range: 2009-12-31 to 2010-12-31

Sharpe Ratio of Fund: 7.69498217453
Sharpe Ratio of $SPX: 0.756512754402
Percent diff:  41.0487699423

Cumulative Return of Fund: 2.08286787
Cumulative Return of $SPX: 0.127827100708
Percent diff:  44.2177866055

Standard Deviation of Fund: 0.00932575927816
Standard Deviation of $SPX: 0.0113715303326
Percent diff:  -4.94212308209

Average Daily Return of Fund: 0.0045205528268
Average Daily Return of $SPX: 0.000541919649169
Percent diff:  39.2953561379
Final Portfolio Value: 30828.6787

and a graph of portfolio performance



