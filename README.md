# Projects
This project uses a helper module called functions1.py, which contains reusable functions for factor analysis, plotting, and performance evaluation. Make sure this file is in the same directory as main.py or adjust the import path accordingly.
The project outputs various performance metrics for each portfolio, including cumulative returns, annualized performance, and risk-adjusted metrics. Additionally, z-score-based portfolios are analyzed for their effectiveness in capturing factor returns.
Key results:
Cumulative returns for momentum and size-based long-short strategies.
Information coefficients (IC) for momentum and size factors.
Alpha-beta regression results showing the risk-adjusted performance of portfolios.
Visualizations of factor performance and portfolio cumulative returns.


Usage Instructions
Main Script (main.py):
This script contains the core logic for downloading stock data, processing the data to calculate factors, constructing long-short portfolios, and evaluating performance.
You can modify the tickers, date range, and portfolio factors directly in the script.

Reusable Functions (functions1.py):
This file contains the custom functions used in the analysis. It includes functions for:
Calculating momentum and size factors.
Making long-short portfolios based on the factors.
Plotting portfolio performance.
Running alpha-beta regressions.
Applying z-score-based long-short strategies.

Customization:
You can modify the thresholds for the z-score strategy, as well as the factors, by adjusting the parameters in the main script.
