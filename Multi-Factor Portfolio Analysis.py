import yfinance as yf
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
import statsmodels as sm
import numpy as np
from scipy.stats import spearmanr
from Reusable_functions import functions1

# Download stock data
tickers = ['AAPL', 'MSFT', 'GOOG', 'TSLA']
data = yf.download(tickers, start="2015-01-01",
                   end="2024-12-31", group_by='tickers')

# Resample to monthly prices and calculate returns
monthly_prices = data.resample('ME').last()
monthly_returns = monthly_prices.pct_change().dropna()

# Calculate momentum and size factors
momentum = functions1.calculate_momentum(monthly_prices)
size = functions1.calculate_size(monthly_prices)

# Extract Close prices properly for size factor analysis
size_close = size.xs('Close', axis=1, level=1)
size_close.index = pd.to_datetime(size_close.index)

# Construct portfolios based on momentum and size
momentum_portfolio = functions1.make_factor_portfolio(
    momentum, monthly_returns)
size_portfolio = functions1.make_factor_portfolio(size, monthly_returns)

# Plot performance of the portfolios
functions1.plot_perf(momentum_portfolio, "Momentum")
functions1.plot_perf(size_portfolio, "Size")

# Calculate information coefficient (IC) for both factors
momentum_ic = functions1.calculate_ic(momentum, monthly_returns)
size_ic = functions1.calculate_ic(size, monthly_returns)

# Plot IC over time
plt.figure(figsize=(10, 4))
momentum_ic.plot(label='Momentum IC')
size_ic.plot(label='Size IC')
plt.title("Information Coefficient (IC) Over Time")
plt.axhline(0, color='black', linestyle='--')
plt.legend()
plt.show()

# Print IC mean and standard deviation
print("Momentum IC Mean:", momentum_ic.mean())
print("Momentum IC Std Dev:", momentum_ic.std())
print("Size IC Mean:", size_ic.mean())
print("Size IC Std Dev:", size_ic.std())

# Download SPY data for comparison
spy_raw = yf.download('SPY', start="2015-01-01", end="2023-12-31")
spy_close = spy_raw[('Close', 'SPY')]
spy_monthly = spy_close.resample('ME').last()
spy_returns = spy_monthly.pct_change().dropna()

# Align data for analysis
common_dates = momentum_portfolio.index.intersection(spy_returns.index)
momentum_portfolio_aligned = momentum_portfolio.loc[common_dates]
size_portfolio_aligned = size_portfolio.loc[common_dates]
spy_returns_aligned = spy_returns.loc[common_dates]

# Run alpha-beta test for both portfolios
functions1.run_alpha_beta_test(
    momentum_portfolio_aligned, spy_returns_aligned, name="Momentum LS")
functions1.run_alpha_beta_test(
    size_portfolio_aligned, spy_returns_aligned, name="Size LS")

# Apply z-score long-short strategy to size factor
print("🎯 size_close.head():\n", size_close.head())
print("🎯 size_close shape:", size_close.shape)
print("🎯 size_close index:", size_close.index[:5])
print("🎯 size_close columns:", size_close.columns)

# Cross-sectional z-score calculation
zscore_factor = size_close.apply(
    lambda row: (row - row.mean()) / row.std(), axis=1)
quantiles = zscore_factor.rank(axis=1, pct=True)

# Long-short signal based on quantiles
longs = quantiles >= 0.7
shorts = quantiles <= 0.3

# Forward returns
returns_df = size_close.pct_change().dropna()
common_dates = size_close.index.intersection(returns_df.index)
zscore_factor = zscore_factor.loc[common_dates]
returns_df = returns_df.loc[common_dates]
longs = longs.loc[common_dates]
shorts = shorts.loc[common_dates]

# Compute factor return
if longs.sum().sum() == 0 or shorts.sum().sum() == 0:
    print("[WARNING] Long/short selections are empty, factor return calculation might be incorrect.")
    factor_return = None
else:
    long_returns = returns_df.where(longs)
    short_returns = returns_df.where(shorts)
    factor_return = long_returns.mean(axis=1) - short_returns.mean(axis=1)

# Plot cumulative return if valid
if factor_return is not None and factor_return.notna().sum() > 0:
    (1 + factor_return).cumprod().plot(title='Z-score Factor Portfolio Cumulative Return')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.show()
else:
    print("[WARNING] Factor return is still all NaN — check if long/short selections are empty.")

# Debug output
print("🎯 Z-score factor describe:\n", zscore_factor.describe())
print("🎯 Quantiles describe:\n", quantiles.describe())
print("🎯 Longs (True/False) sample:\n", longs.head())
print("🎯 Shorts (True/False) sample:\n", shorts.head())

# Apply z-score long-short strategy to size factor with custom thresholds
factor_return, zscore_factor, quantiles, longs, shorts, returns_df = functions1.zscore_long_short_strategy(
    size_close, long_thresh=0.7, short_thresh=0.3)

# Plot and analyze factor return if valid
if factor_return is not None and factor_return.notna().sum() > 0:
    (1 + factor_return).cumprod().plot(title='Cumulative Return of Z-score Factor Portfolio')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.show()
else:
    print("[WARNING] Factor return is still all NaN — check if long/short selections are empty.")

# Performance metrics for each stock based on size factor returns
returns_df_aligned = returns_df.loc[returns_df.index.intersection(
    zscore_factor.index)]
factor_performance = returns_df_aligned.apply(functions1.performance_metrics)
factor_performance_df = pd.DataFrame(
    factor_performance.tolist(), index=returns_df_aligned.columns)

# Plot or print the performance metrics as needed
print("📊 Factor Performance Metrics for Each Stock:\n", factor_performance_df)
