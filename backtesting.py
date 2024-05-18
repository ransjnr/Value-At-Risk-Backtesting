import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import chi2
import plotly.graph_objects as go

data = yf.download('AAPL', start='2015-05-01', end = '2024-05-01')
prices = data['Close']

returns = prices.pct_change().dropna()

returns

confidence_level = 0.99
var_level = np.percentile(returns, 100*(1 - confidence_level))

exceedances = returns < var_level

def kupiec_pof_test(returns, var_level, confidence_level):
    n = len(returns)
    n_fail = exceedances.sum()
    p = 1 - confidence_level
    LR_pof = -2 * np.log(((1-p)**(n-n_fail)) * (p**n_fail)) + 2 * np.log(((1-n_fail/n) ** (n-n_fail))* (n_fail/n) ** n_fail)
    p_value = 1 - chi2.cdf(LR_pof, 1)
    return LR_pof, p_value
    
    
LR_pof, p_value_pof = kupiec_pof_test(returns, var_level, confidence_level)
print("Kupiec POF Test LR Statistic:", LR_pof)
print("P-Value:", p_value_pof)

def christoffersen_test(exceedances):
    n = len(exceedances)
    n_fail  = exceedances.sum()
    clusters = (exceedances[:-1] & exceedances[1:]).sum()
    p_fail = n_fail / n
    p_cluster = clusters / n_fail if n_fail > 0 else 0
    LR_ind = -2 * (np.log((1-p_cluster)**(n_fail - clusters)) + np.log(p_cluster ** clusters)) + 2* (np.log((1 - p_fail) ** (n_fail - clusters)) + np.log(p_fail ** clusters))
    p_value_ind = 1 - chi2.cdf(LR_ind, 1)
    return LR_ind, p_value_ind

LR_pof, p_value_ind = christoffersen_test(exceedances)
print("Christoffersen's Test LR Statistic:", LR_pof)
print("P-Value:", p_value_ind)

def conditional_coverage_test(LR_pof, LR_ind):
    LR_cc = LR_pof + LR_ind
    p_value_cc = 1 - chi2.cdf(LR_cc, 2)
    return LR_cc, p_value_cc

LR_cc, p_value_cc = conditional_coverage_test(LR_pof, 2)
print("Conditional Coverage Test LR Statistic:", LR_cc)
print("p-value:", p_value_cc)

fig = go.Figure()
fig.add_trace(go.Scatter(x=returns.index, y=returns, mode = 'lines', name = 'Daily_Returns'))
fig.add_trace(go.Scatter(x=returns[exceedances].index, y=returns[exceedances],mode='markers', name = 'Exceedances', marker = dict(color='red')))
fig.update_layout(title = "VaR Exceedances Visualization", xaxis_title = "Date", yaxis_title = "Returns", legend_title ="Legend")

fig.show()