import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import chi2
import plotly.graph_objects as go

# Download historical data for Apple (AAPL) from Yahoo Finance
data = yf.download('AAPL', start='2015-05-01', end='2024-05-01')
prices = data['Close']

# Calculate daily returns
returns = prices.pct_change().dropna()

returns

# Set confidence level for VaR
confidence_level = 0.99
# Calculate the VaR level based on the confidence level
var_level = np.percentile(returns, 100*(1 - confidence_level))

# Identify exceedances (returns below VaR level)
exceedances = returns < var_level

# Kupiec's Proportion of Failures (POF) Test
def kupiec_pof_test(returns, var_level, confidence_level):
    n = len(returns)  # Total number of returns
    n_fail = exceedances.sum()  # Number of exceedances (failures)
    p = 1 - confidence_level  # Expected failure probability
    # Likelihood Ratio statistic for POF test
    LR_pof = -2 * np.log(((1-p)**(n-n_fail)) * (p**n_fail)) + 2 * np.log(((1-n_fail/n) ** (n-n_fail)) * (n_fail/n) ** n_fail)
    p_value = 1 - chi2.cdf(LR_pof, 1)  # P-value for the test
    return LR_pof, p_value

# Calculate Kupiec POF test statistic and p-value
LR_pof, p_value_pof = kupiec_pof_test(returns, var_level, confidence_level)
print("Kupiec POF Test LR Statistic:", LR_pof)
print("P-Value:", p_value_pof)

# Christoffersen's Test for Independence of Exceedances
def christoffersen_test(exceedances):
    n = len(exceedances)  # Total number of observations
    n_fail = exceedances.sum()  # Number of exceedances
    clusters = (exceedances[:-1] & exceedances[1:]).sum()  # Number of clustered exceedances
    p_fail = n_fail / n  # Probability of exceedance
    p_cluster = clusters / n_fail if n_fail > 0 else 0  # Probability of clustering given an exceedance
    # Likelihood Ratio statistic for Independence test
    LR_ind = -2 * (np.log((1-p_cluster)**(n_fail - clusters)) + np.log(p_cluster ** clusters)) + 2 * (np.log((1 - p_fail) ** (n_fail - clusters)) + np.log(p_fail ** clusters))
    p_value_ind = 1 - chi2.cdf(LR_ind, 1)  # P-value for the test
    return LR_ind, p_value_ind

# Calculate Christoffersen's test statistic and p-value
LR_ind, p_value_ind = christoffersen_test(exceedances)
print("Christoffersen's Test LR Statistic:", LR_ind)
print("P-Value:", p_value_ind)

# Conditional Coverage Test combining POF and Independence tests
def conditional_coverage_test(LR_pof, LR_ind):
    LR_cc = LR_pof + LR_ind  # Combined Likelihood Ratio statistic
    p_value_cc = 1 - chi2.cdf(LR_cc, 2)  # P-value for the combined test
    return LR_cc, p_value_cc

# Calculate Conditional Coverage test statistic and p-value
LR_cc, p_value_cc = conditional_coverage_test(LR_pof, LR_ind)
print("Conditional Coverage Test LR Statistic:", LR_cc)
print("p-value:", p_value_cc)

# Visualization of returns and VaR exceedances
fig = go.Figure()
fig.add_trace(go.Scatter(x=returns.index, y=returns, mode='lines', name='Daily_Returns'))
fig.add_trace(go.Scatter(x=returns[exceedances].index, y=returns[exceedances], mode='markers', name='Exceedances', marker=dict(color='red')))
fig.update_layout(title="VaR Exceedances Visualization", xaxis_title="Date", yaxis_title="Returns", legend_title="Legend")

fig.show()
