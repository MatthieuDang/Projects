import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from scipy.stats import spearmanr


def calculate_momentum(prices):
    mom_12m = prices.pct_change(12)
    mom_1m = prices.pct_change(1)
    return (mom_12m - mom_1m).shift(1).dropna()


def calculate_size(prices):
    return -prices.shift(1)


def make_factor_portfolio(factor_df, returns_df, quantile=0.2):
    long_returns = []
    short_returns = []
    valid_dates = []

    for date in factor_df.index[1:]:
        scores_raw = factor_df.loc[date]
        rets_raw = returns_df.loc[date]
        common = scores_raw.index.intersection(rets_raw.index)
        scores = scores_raw.loc[common]
        rets = rets_raw.loc[common]

        # Debugging step: check if any values are NaN
        if scores.isna().any() or rets.isna().any():
            print(f"[SKIPPED] {date}: NaN found in scores or returns.")
            long_returns.append(np.nan)
            short_returns.append(np.nan)
            continue

        if len(scores) < 3:
            print(f"[SKIPPED] {date}: Not enough data points ({len(scores)}).")
            long_returns.append(np.nan)
            short_returns.append(np.nan)
            continue

        quantile_cutoff = int(len(scores) * quantile)
        ranked = scores.sort_values()
        short = ranked[:quantile_cutoff].index
        long = ranked[-quantile_cutoff:].index

        long_ret = rets.loc[long].mean()
        short_ret = rets.loc[short].mean()
        spread = long_ret - short_ret

        print(
            f"[OK] {date}: Long={long_ret:.4f}, Short={short_ret:.4f}, Spread={spread:.4f}")

        long_returns.append(long_ret)
        short_returns.append(short_ret)
        valid_dates.append(date)

    portfolio = pd.Series(np.array(long_returns) -
                          np.array(short_returns), index=valid_dates)
    return portfolio


def plot_perf(series, title):
    if series.isna().all():
        print(f"[WARNING] {title} series is all NaN — nothing to plot.")
        return
    if (series == 0).all():
        print(f"[WARNING] {title} series is all zeros — nothing to plot.")
        return

    cumulative = (1 + series.dropna()).cumprod()
    if cumulative.empty:
        print(f"[WARNING] {title} cumulative return is empty.")
        return

    cumulative.plot(figsize=(10, 5))
    plt.title(f"Cumulative Return: {title}")
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.grid(True)
    plt.show()


def calculate_ic(factor_df, returns_df):
    ic_series = []
    for i in range(1, len(factor_df) - 2):
        factor = factor_df.iloc[i]
        future_ret = returns_df.iloc[i + 1]

        common = factor.index.intersection(future_ret.index)
        factor = factor.loc[common]
        future_ret = future_ret.loc[common]

        if len(factor) > 5:  # Need enough data points
            ic, _ = spearmanr(factor, future_ret)
            ic_series.append(ic)
        else:
            ic_series.append(np.nan)

    ic_series = pd.Series(
        ic_series, index=factor_df.index[2: 2 + len(ic_series)])
    return ic_series


def run_alpha_beta_test(factor_returns, market_returns, name="Factor"):
    df = pd.concat([factor_returns, market_returns], axis=1).dropna()
    X = sm.add_constant(market_returns)
    y = factor_returns
    model = sm.OLS(y, X).fit()
    print(f"\n{name} Portfolio Regression on Market:")
    print(model.summary())
    return model


def performance_metrics(factor_returns):

    annualized_return = (1 + factor_returns.mean()) ** 12 - 1

    volatility = factor_returns.std() * np.sqrt(12)

    sharpe_ratio = annualized_return / volatility

    cumulative_returns = (1 + factor_returns).cumprod() - 1
    max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()

    return {
        'Annualized Return': annualized_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown
    }


def zscore_long_short_strategy(price_df, long_thresh=0.7, short_thresh=0.3):
    # Compute the Z-scores for the factor
    zscore_factor = (price_df - price_df.mean()) / price_df.std()

    # Create quantiles for the long-short positions
    quantiles = zscore_factor.quantile([short_thresh, long_thresh])

    # Create long/short positions based on the quantiles
    longs = zscore_factor > quantiles.iloc[1]  # Long if above 70th percentile
    # Short if below 30th percentile
    shorts = zscore_factor < quantiles.iloc[0]

    # Calculate factor return: long minus short positions
    factor_return = (longs * price_df.pct_change()).mean(axis=1) - \
        (shorts * price_df.pct_change()).mean(axis=1)

    # Ensure that factor_return is returned properly in the tuple
    return factor_return, zscore_factor, quantiles, longs, shorts, price_df.pct_change()
